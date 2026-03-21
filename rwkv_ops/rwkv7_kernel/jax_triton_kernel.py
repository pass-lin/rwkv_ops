"""
JAX 版 RWKV7 Triton Kernel 封装
利用 jax-triton 直接调用 triton_kernel.py 中的算子，并支持 SPMD 切分
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import jax_triton as jt
import jax.tree_util as jtu
from typing import Optional, Tuple, Union

# 引入 Triton 核心算子
from .triton_kernel import (
    rwkv7_fwd_kernel,
    rwkv7_bwd_kernel,
    rwkv7_fwd_kernel_with_mask,
    rwkv7_bwd_kernel_with_mask,
)

# 引入自定义分区器 (适配 JAX 新版 Shardy 引擎)
from jax.experimental.custom_partitioning import custom_partitioning

CHUNK_LEN = 16

# =========================================================================
# SPMD 切分规则 (Einsum 风格)
# b=Batch, n=Head, t=Time, h=HeadDim1, m=HeadDim2, c=Chunk
# =========================================================================
# 无 Mask 规则
FWD_RULE = "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n h m -> b n t h, b n t h, b n c h m"
BWD_RULE = "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n c h m, b n h m -> b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n h m"

# 带 Mask 规则
FWD_MASK_RULE = "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n h m, b t -> b n t h, b n t h, b n c h m"
BWD_MASK_RULE = "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b t, b n t h, b n t h, b n c h m, b n h m -> b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n h m"


def _fwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]  # 与输入对齐
    return (qs, qs, qs)


def _bwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    return (qs, qs, qs, qs, qs, qs, qs)


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


# =========================================================================
# 无 Mask 的 JAX-Triton Launcher
# =========================================================================
def _wkv7_fwd_triton_call(r, w, k, v, a, b, h0):
    B, N, T, H = r.shape
    dtype = r.dtype
    chunk_num = T // CHUNK_LEN

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # OUT
        jax.ShapeDtypeStruct((B, N, T, H), jnp.float32),  # SA_OUT
        jax.ShapeDtypeStruct((B, N, chunk_num, H, H), jnp.float32),  # STATE_CHKP
    ]

    grid = lambda meta: ((B + meta["MINI_BSZ"] - 1) // meta["MINI_BSZ"], N)

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
        CHUNK_LEN=CHUNK_LEN,
    )
    return out, sa_out, state_chkp


@custom_partitioning
def _wkv7_fwd_spmd(r, w, k, v, a, b, h0):
    return _wkv7_fwd_triton_call(r, w, k, v, a, b, h0)


_wkv7_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=_create_partition(_wkv7_fwd_triton_call),
)


def _wkv7_bwd_triton_call(r, w, k, v, a, b, dy, sa, state_chkp, dht):
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

    grid = lambda meta: ((B + meta["MINI_BSZ"] - 1) // meta["MINI_BSZ"], N)

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
        CHUNK_LEN=CHUNK_LEN,
    )
    return dr, dw, dk, dv, da, db, dh0


@custom_partitioning
def _wkv7_bwd_spmd(r, w, k, v, a, b, dy, sa, state_chkp, dht):
    return _wkv7_bwd_triton_call(r, w, k, v, a, b, dy, sa, state_chkp, dht)


_wkv7_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=_create_partition(_wkv7_bwd_triton_call),
)


@jax.custom_vjp
def rwkv7_kernel_triton(r, w, k, v, a, b, h0):
    out, sa_out, state_chkp = _wkv7_fwd_spmd(r, w, k, v, a, b, h0)
    final_state = state_chkp[:, :, -1, :, :]
    return out, final_state


def _fwd(r, w, k, v, a, b, h0):
    out, sa_out, state_chkp = _wkv7_fwd_spmd(r, w, k, v, a, b, h0)
    final_state = state_chkp[:, :, -1, :, :]
    return (out, final_state), (r, w, k, v, a, b, sa_out, state_chkp)


def _bwd(res, grads):
    r, w, k, v, a, b, sa_out, state_chkp = res
    dy, dht = grads
    dy = jnp.asarray(dy, jnp.bfloat16)
    if dht is None:
        B, N, T, H = r.shape
        dht = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    dr, dw, dk, dv, da, db, dh0 = _wkv7_bwd_spmd(
        r, w, k, v, a, b, dy, sa_out, state_chkp, dht
    )
    return dr, dw, dk, dv, da, db, dh0


rwkv7_kernel_triton.defvjp(_fwd, _bwd)


# =========================================================================
# 带 Mask 的 JAX-Triton Launcher
# =========================================================================
def _wkv7_fwd_with_mask_triton_call(r, w, k, v, a, b, h0, mask):
    B, N, T, H = r.shape
    dtype = r.dtype
    chunk_num = T // CHUNK_LEN

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, H), dtype),
        jax.ShapeDtypeStruct((B, N, T, H), jnp.float32),
        jax.ShapeDtypeStruct((B, N, chunk_num, H, H), jnp.float32),
    ]

    grid = lambda meta: ((B + meta["MINI_BSZ"] - 1) // meta["MINI_BSZ"], N)

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
        CHUNK_LEN=CHUNK_LEN,
    )
    return out, sa_out, state_chkp


@custom_partitioning
def _wkv7_fwd_with_mask_spmd(r, w, k, v, a, b, h0, mask):
    return _wkv7_fwd_with_mask_triton_call(r, w, k, v, a, b, h0, mask)


_wkv7_fwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_MASK_RULE,
    partition=_create_partition(_wkv7_fwd_with_mask_triton_call),
)


def _wkv7_bwd_with_mask_triton_call(r, w, k, v, a, b, mask, dy, sa, state_chkp, dht):
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

    grid = lambda meta: ((B + meta["MINI_BSZ"] - 1) // meta["MINI_BSZ"], N)

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
        CHUNK_LEN=CHUNK_LEN,
    )
    return dr, dw, dk, dv, da, db, dh0


@custom_partitioning
def _wkv7_bwd_with_mask_spmd(r, w, k, v, a, b, mask, dy, sa, state_chkp, dht):
    return _wkv7_bwd_with_mask_triton_call(
        r, w, k, v, a, b, mask, dy, sa, state_chkp, dht
    )


_wkv7_bwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_MASK_RULE,
    partition=_create_partition(_wkv7_bwd_with_mask_triton_call),
)


@jax.custom_vjp
def rwkv7_kernel_with_mask_triton(r, w, k, v, a, b, h0, mask):
    out, sa_out, state_chkp = _wkv7_fwd_with_mask_spmd(r, w, k, v, a, b, h0, mask)
    final_state = state_chkp[:, :, -1, :, :]
    return out, final_state


def _fwd_with_mask(r, w, k, v, a, b, h0, mask):
    out, sa_out, state_chkp = _wkv7_fwd_with_mask_spmd(r, w, k, v, a, b, h0, mask)
    final_state = state_chkp[:, :, -1, :, :]
    return (out, final_state), (r, w, k, v, a, b, mask, sa_out, state_chkp)


def _bwd_with_mask(res, grads):
    r, w, k, v, a, b, mask, sa_out, state_chkp = res
    dy, dht = grads
    dy = jnp.asarray(dy, jnp.bfloat16)
    if dht is None:
        B, N, T, H = r.shape
        dht = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    dr, dw, dk, dv, da, db, dh0 = _wkv7_bwd_with_mask_spmd(
        r, w, k, v, a, b, mask, dy, sa_out, state_chkp, dht
    )
    return dr, dw, dk, dv, da, db, dh0, None


rwkv7_kernel_with_mask_triton.defvjp(_fwd_with_mask, _bwd_with_mask)


# =========================================================================
# 对外 API 暴露 (与 JAX/Torch CUDA 版本兼容对齐)
# =========================================================================
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
    mask: Optional[jnp.ndarray] = None,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:

    # 统一转换到 Head-First [B, N, T, H]
    r = _transpose_head(r, head_first)
    w = _transpose_head(w, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    a = _transpose_head(a, head_first)
    b = _transpose_head(b, head_first)

    B, N, T, H = r.shape
    if T % CHUNK_LEN != 0:
        raise ValueError(
            f"Triton kernel requires sequence length T={T} to be divisible by {CHUNK_LEN}"
        )

    # 准备初始状态
    if initial_state is None:
        h0 = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        h0 = jnp.asarray(initial_state, dtype=jnp.float32)

    # 路由调用 Mask 还是 Non-Mask
    if mask is None:
        out, last_state = rwkv7_kernel_triton(r, w, k, v, a, b, h0)
    else:
        if mask.shape != (B, T) and mask.shape != (B, T, 1, 1):
            raise ValueError(
                f"Mask shape must be (B, T) or (B, T, 1, 1), got {mask.shape}"
            )
        mask = jnp.asarray(mask, dtype=jnp.float32).reshape(B, T)
        out, last_state = rwkv7_kernel_with_mask_triton(r, w, k, v, a, b, h0, mask)

    out = jnp.transpose(out, (0, 2, 1, 3))

    if output_final_state:
        return out, last_state
    return out
