"""JAX 版 RWKV7 Triton kernel 封装。"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import jax_triton as jt
import jax.tree_util as jtu
from typing import Optional, Tuple, Union

from .triton_kernel import (
    rwkv7_fwd_kernel,
    rwkv7_bwd_kernel,
    rwkv7_fwd_kernel_with_mask,
    rwkv7_bwd_kernel_with_mask,
)

from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

#  SPMD 切分规则（Einsum 风格）
# b=Batch, n=Head, t=Time, h=HeadDim1, m=HeadDim2, c=Chunk
# 无 Mask 规则
FWD_RULE = "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n h m -> b n t h, b n t h, b n c h m"
BWD_RULE = "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n c h m, b n h m -> b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n h m"

# 带 Mask 规则
FWD_MASK_RULE = "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n h m, b t -> b n t h, b n t h, b n c h m"
BWD_MASK_RULE = "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b t, b n t h, b n t h, b n c h m, b n h m -> b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n h m"


def _q_spec(qs):
    spec = getattr(qs, "spec", None)
    if spec is None or len(spec) != 4:
        return None
    return spec


def _sharding_like_q(qs):
    """为 y / sa (B, N, T, H) 构造与输入一致的 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(*spec))


def _sharding_for_state(qs):
    """为 State checkpoint (B, N, C, H, H) 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(
        qs.mesh, PartitionSpec(spec[0], spec[1], None, spec[3], spec[3])
    )


def _sharding_for_final_state(qs):
    """为最终 State / dh0 (B, N, H, H) 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], spec[3], spec[3]))


def _fwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    q_like = _sharding_like_q(qs)
    return (q_like, q_like, _sharding_for_state(qs))


def _bwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    q_like = _sharding_like_q(qs)
    h0_like = _sharding_for_final_state(qs)
    return (q_like, q_like, q_like, q_like, q_like, q_like, h0_like)


def _create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)

        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition


def _transpose_head(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.bfloat16)
    if not head_first:
        # [B, T, N, H] -> [B, N, T, H]
        return jnp.transpose(x, (0, 2, 1, 3))
    return x


#  无 Mask 的 JAX-Triton Launcher
def _wkv7_fwd_triton_call(r, w, k, v, a, b, h0, chunk_size: int):
    B, N, T, H = r.shape
    dtype = r.dtype
    chunk_num = T // chunk_size

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # OUT
        jax.ShapeDtypeStruct((B, N, T, H), jnp.float32),  # SA_OUT
        jax.ShapeDtypeStruct((B, N, chunk_num, H, H), jnp.float32),  # STATE_CHKP
    ]

    def grid(meta):
        return ((B + meta["MINI_BSZ"] - 1) // meta["MINI_BSZ"], N)

    out, sa_out, state_chkp = jt.triton_call(
        r,
        w,
        k,
        v,
        a,
        b,
        h0,
        B,
        N,
        T,
        kernel=rwkv7_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,
        CHUNK_LEN=chunk_size,
    )
    return out, sa_out, state_chkp


def _wkv7_bwd_triton_call(r, w, k, v, a, b, dy, sa, state_chkp, dht, chunk_size: int):
    B, N, T, H = r.shape
    dtype = r.dtype

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DR
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DW
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DK
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DV
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DA
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DB
        jax.ShapeDtypeStruct((B, N, H, H), jnp.float32),  # DH0
    ]

    def grid(meta):
        return ((B + meta["MINI_BSZ"] - 1) // meta["MINI_BSZ"], N)

    dr, dw, dk, dv, da, db, dh0 = jt.triton_call(
        r,
        w,
        k,
        v,
        a,
        b,
        sa,
        state_chkp,
        B,
        N,
        T,
        dy,
        dht,
        kernel=rwkv7_bwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,
        CHUNK_LEN=chunk_size,
    )
    return dr, dw, dk, dv, da, db, dh0


@custom_partitioning
def _wkv7_fwd_spmd(r, w, k, v, a, b, h0, chunk_size: int):
    return _wkv7_fwd_triton_call(r, w, k, v, a, b, h0, chunk_size)


_wkv7_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=_create_partition(_wkv7_fwd_triton_call),
)


@custom_partitioning
def _wkv7_bwd_spmd(r, w, k, v, a, b, dy, sa, state_chkp, dht, chunk_size: int):
    return _wkv7_bwd_triton_call(r, w, k, v, a, b, dy, sa, state_chkp, dht, chunk_size)


_wkv7_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=_create_partition(_wkv7_bwd_triton_call),
)


@jax.custom_vjp
def rwkv7_kernel_triton(r, w, k, v, a, b, h0, chunk_size: int):
    out, sa_out, state_chkp = _wkv7_fwd_spmd(r, w, k, v, a, b, h0, chunk_size)
    final_state = state_chkp[:, :, -1, :, :]
    return out, final_state


def _fwd(r, w, k, v, a, b, h0, chunk_size: int):
    out, sa_out, state_chkp = _wkv7_fwd_spmd(r, w, k, v, a, b, h0, chunk_size)
    final_state = state_chkp[:, :, -1, :, :]
    return (out, final_state), (r, w, k, v, a, b, sa_out, state_chkp, chunk_size)


def _bwd(res, grads):
    r, w, k, v, a, b, sa_out, state_chkp, chunk_size = res
    dy, dht = grads
    dy = jnp.asarray(dy, jnp.bfloat16)
    if dht is None:
        B, N, T, H = r.shape
        dht = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    dr, dw, dk, dv, da, db, dh0 = _wkv7_bwd_spmd(
        r, w, k, v, a, b, dy, sa_out, state_chkp, dht, chunk_size
    )
    return dr, dw, dk, dv, da, db, dh0


rwkv7_kernel_triton.defvjp(_fwd, _bwd)


#  带 Mask 的 JAX-Triton Launcher
def _wkv7_fwd_with_mask_triton_call(r, w, k, v, a, b, h0, mask, chunk_size: int):
    B, N, T, H = r.shape
    dtype = r.dtype
    chunk_num = T // chunk_size

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, H), dtype),
        jax.ShapeDtypeStruct((B, N, T, H), jnp.float32),
        jax.ShapeDtypeStruct((B, N, chunk_num, H, H), jnp.float32),
    ]

    def grid(meta):
        return ((B + meta["MINI_BSZ"] - 1) // meta["MINI_BSZ"], N)

    out, sa_out, state_chkp = jt.triton_call(
        r,
        w,
        k,
        v,
        a,
        b,
        mask,
        h0,
        B,
        N,
        T,
        kernel=rwkv7_fwd_kernel_with_mask,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,
        CHUNK_LEN=chunk_size,
    )
    return out, sa_out, state_chkp


@custom_partitioning
def _wkv7_fwd_with_mask_spmd(r, w, k, v, a, b, h0, mask, chunk_size: int):
    return _wkv7_fwd_with_mask_triton_call(r, w, k, v, a, b, h0, mask, chunk_size)


_wkv7_fwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_MASK_RULE,
    partition=_create_partition(_wkv7_fwd_with_mask_triton_call),
)


def _wkv7_bwd_with_mask_triton_call(
    r, w, k, v, a, b, mask, dy, sa, state_chkp, dht, chunk_size: int
):
    B, N, T, H = r.shape
    dtype = r.dtype

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DR
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DW
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DK
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DV
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DA
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DB
        jax.ShapeDtypeStruct((B, N, H, H), jnp.float32),  # DH0
    ]

    def grid(meta):
        return ((B + meta["MINI_BSZ"] - 1) // meta["MINI_BSZ"], N)

    dr, dw, dk, dv, da, db, dh0 = jt.triton_call(
        r,
        w,
        k,
        v,
        a,
        b,
        mask,
        sa,
        state_chkp,
        B,
        N,
        T,
        dy,
        dht,
        kernel=rwkv7_bwd_kernel_with_mask,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,
        CHUNK_LEN=chunk_size,
    )
    return dr, dw, dk, dv, da, db, dh0


@custom_partitioning
def _wkv7_bwd_with_mask_spmd(
    r, w, k, v, a, b, mask, dy, sa, state_chkp, dht, chunk_size: int
):
    return _wkv7_bwd_with_mask_triton_call(
        r, w, k, v, a, b, mask, dy, sa, state_chkp, dht, chunk_size
    )


_wkv7_bwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_MASK_RULE,
    partition=_create_partition(_wkv7_bwd_with_mask_triton_call),
)


@jax.custom_vjp
def rwkv7_kernel_with_mask_triton(r, w, k, v, a, b, h0, mask, chunk_size: int):
    out, sa_out, state_chkp = _wkv7_fwd_with_mask_spmd(
        r, w, k, v, a, b, h0, mask, chunk_size
    )
    final_state = state_chkp[:, :, -1, :, :]
    return out, final_state


def _fwd_with_mask(r, w, k, v, a, b, h0, mask, chunk_size: int):
    out, sa_out, state_chkp = _wkv7_fwd_with_mask_spmd(
        r, w, k, v, a, b, h0, mask, chunk_size
    )
    final_state = state_chkp[:, :, -1, :, :]
    return (out, final_state), (
        r,
        w,
        k,
        v,
        a,
        b,
        mask,
        sa_out,
        state_chkp,
        chunk_size,
    )


def _bwd_with_mask(res, grads):
    r, w, k, v, a, b, mask, sa_out, state_chkp, chunk_size = res
    dy, dht = grads
    dy = jnp.asarray(dy, jnp.bfloat16)
    if dht is None:
        B, N, T, H = r.shape
        dht = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    dr, dw, dk, dv, da, db, dh0 = _wkv7_bwd_with_mask_spmd(
        r, w, k, v, a, b, mask, dy, sa_out, state_chkp, dht, chunk_size
    )
    return dr, dw, dk, dv, da, db, dh0, None


rwkv7_kernel_with_mask_triton.defvjp(_fwd_with_mask, _bwd_with_mask)


#  对外 API
def generalized_delta_rule(
    r: jnp.ndarray,
    w: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = False,
    chunk_size: int = 16,
    mask: Optional[jnp.ndarray] = None,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """RWKV-7 chunkwise 训练算子（JAX Triton 实现）。

    Args:
        r, w, k, v, a, b: [B, T, H, K]，bfloat16。T 必须被 chunk_size 整除。
        initial_state: [B, H, K, K] 或 [1, H, K, K]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, K]）。
        chunk_size: int，chunk 长度，必须整除序列长度。
        mask: [B, T] 或 [B, T, 1, 1]，float32，1 表示更新状态、0 表示冻结状态。

    Returns:
        out: [B, T, H, K]，与输入同 dtype。
        final_state: [B, H, K, K]，float32；仅当 output_final_state=True 时返回。

    Raises:
        ValueError: T 不被 chunk_size 整除，或 mask 形状不匹配。
    """
    dtype = r.dtype
    # 统一转换到 Head-First [B, N, T, H]
    r = _transpose_head(r, head_first)
    w = _transpose_head(w, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    a = _transpose_head(a, head_first)
    b = _transpose_head(b, head_first)

    B, N, T, H = r.shape
    if T % chunk_size != 0:
        raise ValueError(
            f"Triton kernel requires sequence length T={T} to be divisible by {chunk_size}"
        )

    # 准备初始状态
    if initial_state is None:
        h0 = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        h0 = jnp.asarray(initial_state, dtype=jnp.float32)

    # 路由调用 Mask 还是 Non-Mask
    if mask is None:
        out, last_state = rwkv7_kernel_triton(r, w, k, v, a, b, h0, chunk_size)
    else:
        if mask.shape != (B, T) and mask.shape != (B, T, 1, 1):
            raise ValueError(
                f"Mask shape must be (B, T) or (B, T, 1, 1), got {mask.shape}"
            )
        mask = jnp.asarray(mask, dtype=jnp.float32).reshape(B, T)
        out, last_state = rwkv7_kernel_with_mask_triton(
            r, w, k, v, a, b, h0, mask, chunk_size
        )

    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if output_final_state:
        return out, last_state
    return out
