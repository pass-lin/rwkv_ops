"""JAX 版 RWKV7-SANE Triton kernel 封装。"""

from __future__ import annotations

import functools
import warnings
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import jax_triton as jt

from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

from .triton_kernel import (
    rwkv7_sane_bwd_kernel,
    rwkv7_sane_bwd_kernel_with_mask,
    rwkv7_sane_fwd_kernel,
    rwkv7_sane_fwd_kernel_with_mask,
)

#  SPMD 切分规则（Einsum 风格）
# b=Batch, n=Head, t=Time, h=HeadDim, c=Chunk
FWD_RULE = (
    "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n c, b n h h -> "
    "b n t h, b n t h, b n c h h"
)
BWD_RULE = (
    "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n c h h, "
    "b n c, b n t h, b n h h -> b n t h, b n t h, b n t h, b n t h, b n t h, "
    "b n t h, b n c, b n h h"
)

FWD_MASK_RULE = (
    "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n c, b c, b n h h -> "
    "b n t h, b n t h, b n c h h"
)
BWD_MASK_RULE = (
    "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n c h h, "
    "b n c, b c, b n t h, b n h h -> b n t h, b n t h, b n t h, b n t h, b n t h, "
    "b n t h, b n c, b n h h"
)


def _q_spec(qs):
    spec = getattr(qs, "spec", None)
    if spec is None or len(spec) != 4:
        return None
    return spec


def _sharding_like_q(qs):
    """为输出构造与输入 q 同形的 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(*spec))


def _sharding_for_state(qs):
    """为 state checkpoint [B, N, C, H, H] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(
        qs.mesh, PartitionSpec(spec[0], spec[1], None, spec[3], spec[3])
    )


def _sharding_for_final_state(qs):
    """为最终 State / dh0 [B, N, H, H] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], spec[3], spec[3]))


def _sharding_for_tau(qs):
    """为 tau / dtau [B, N, C] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], None))


def _fwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    return (
        _sharding_like_q(qs),
        _sharding_like_q(qs),
        _sharding_for_state(qs),
    )


def _bwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    q_like = _sharding_like_q(qs)
    tau_like = _sharding_for_tau(qs)
    h0_like = _sharding_for_final_state(qs)
    return (
        q_like,
        q_like,
        q_like,
        q_like,
        q_like,
        q_like,
        tau_like,
        h0_like,
    )


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


def _transpose_tau(tau: jnp.ndarray) -> jnp.ndarray:
    """tau 公共接口始终为 [B, T//16, N]；需要转成 head-first [B, N, T//16]。"""
    tau = jnp.asarray(tau, dtype=jnp.float32)
    # [B, T//16, N] -> [B, N, T//16]
    return jnp.transpose(tau, (0, 2, 1))


def _apply_sane_to_final_state(
    state: jnp.ndarray,
    tau: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """对最终 state 应用 State Anomaly Neutralization。

    Args:
        state: [B, N, H, H], float32。
        tau: [B, N, C], float32。
        mask: [B, C], float32（可选）。

    Returns:
        [B, N, H, H], float32。
    """
    last_tau = tau[:, :, -1][:, :, None, None]
    tau_safe = jnp.maximum(last_tau, 1e-6)
    sane_state = last_tau * jnp.tanh(state / tau_safe)
    if mask is None:
        return sane_state
    last_mask = mask[:, -1][:, None, None, None]
    return jnp.where(last_mask > 0, sane_state, state)


#  无 mask 的 JAX-Triton launcher
def _wkv7_sane_fwd_triton_call(r, w, k, v, a, b, tau, h0, chunk_size: int):
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
        tau,
        h0,
        B,
        N,
        T,
        kernel=rwkv7_sane_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,
        CHUNK_LEN=chunk_size,
    )
    return out, sa_out, state_chkp


@functools.partial(custom_partitioning, static_argnums=(8,))
def _wkv7_sane_fwd_spmd(r, w, k, v, a, b, tau, h0, chunk_size: int):
    return _wkv7_sane_fwd_triton_call(r, w, k, v, a, b, tau, h0, chunk_size)


_wkv7_sane_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=_create_partition(_wkv7_sane_fwd_triton_call),
)


def _wkv7_sane_bwd_triton_call(
    r, w, k, v, a, b, sa, state_chkp, tau, dy, dht, chunk_size: int
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
        jax.ShapeDtypeStruct((B, N, T // chunk_size), jnp.float32),  # DTAU
        jax.ShapeDtypeStruct((B, N, H, H), jnp.float32),  # DH0
    ]

    def grid(meta):
        return ((B + meta["MINI_BSZ"] - 1) // meta["MINI_BSZ"], N)

    dr, dw, dk, dv, da, db, dtau, dh0 = jt.triton_call(
        r,
        w,
        k,
        v,
        a,
        b,
        sa,
        state_chkp,
        tau,
        B,
        N,
        T,
        dy,
        dht,
        kernel=rwkv7_sane_bwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,
        CHUNK_LEN=chunk_size,
    )
    return dr, dw, dk, dv, da, db, dtau, dh0


@functools.partial(custom_partitioning, static_argnums=(11,))
def _wkv7_sane_bwd_spmd(
    r, w, k, v, a, b, sa, state_chkp, tau, dy, dht, chunk_size: int
):
    return _wkv7_sane_bwd_triton_call(
        r, w, k, v, a, b, sa, state_chkp, tau, dy, dht, chunk_size
    )


_wkv7_sane_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=_create_partition(_wkv7_sane_bwd_triton_call),
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(8,))
def rwkv7_sane_kernel_triton(r, w, k, v, a, b, tau, h0, chunk_size: int):
    """无 mask JAX-Triton 训练 kernel 公开入口。"""
    out, sa_out, state_chkp = _wkv7_sane_fwd_spmd(r, w, k, v, a, b, tau, h0, chunk_size)
    final_state = _apply_sane_to_final_state(state_chkp[:, :, -1, :, :], tau)
    return out, final_state


def _fwd(r, w, k, v, a, b, tau, h0, chunk_size: int):
    out, sa_out, state_chkp = _wkv7_sane_fwd_spmd(r, w, k, v, a, b, tau, h0, chunk_size)
    final_state = _apply_sane_to_final_state(state_chkp[:, :, -1, :, :], tau)
    return (out, final_state), (r, w, k, v, a, b, tau, sa_out, state_chkp)


def _bwd(chunk_size, res, grads):
    r, w, k, v, a, b, tau, sa_out, state_chkp = res
    dy, dht = grads
    dy = jnp.asarray(dy, jnp.bfloat16)
    if dht is None:
        B, N, T, H = r.shape
        dht = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    dr, dw, dk, dv, da, db, dtau, dh0 = _wkv7_sane_bwd_spmd(
        r, w, k, v, a, b, sa_out, state_chkp, tau, dy, dht, chunk_size
    )
    return dr, dw, dk, dv, da, db, dtau, dh0


rwkv7_sane_kernel_triton.defvjp(_fwd, _bwd)


#  带 mask 的 JAX-Triton launcher
def _wkv7_sane_fwd_with_mask_triton_call(
    r, w, k, v, a, b, tau, mask, h0, chunk_size: int
):
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
        tau,
        mask,
        h0,
        B,
        N,
        T,
        kernel=rwkv7_sane_fwd_kernel_with_mask,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,
        CHUNK_LEN=chunk_size,
    )
    return out, sa_out, state_chkp


@functools.partial(custom_partitioning, static_argnums=(9,))
def _wkv7_sane_fwd_with_mask_spmd(r, w, k, v, a, b, tau, mask, h0, chunk_size: int):
    return _wkv7_sane_fwd_with_mask_triton_call(
        r, w, k, v, a, b, tau, mask, h0, chunk_size
    )


_wkv7_sane_fwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_MASK_RULE,
    partition=_create_partition(_wkv7_sane_fwd_with_mask_triton_call),
)


def _wkv7_sane_bwd_with_mask_triton_call(
    r, w, k, v, a, b, sa, state_chkp, tau, mask, dy, dht, chunk_size: int
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
        jax.ShapeDtypeStruct((B, N, T // chunk_size), jnp.float32),  # DTAU
        jax.ShapeDtypeStruct((B, N, H, H), jnp.float32),  # DH0
    ]

    def grid(meta):
        return ((B + meta["MINI_BSZ"] - 1) // meta["MINI_BSZ"], N)

    dr, dw, dk, dv, da, db, dtau, dh0 = jt.triton_call(
        r,
        w,
        k,
        v,
        a,
        b,
        sa,
        state_chkp,
        tau,
        mask,
        B,
        N,
        T,
        dy,
        dht,
        kernel=rwkv7_sane_bwd_kernel_with_mask,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,
        CHUNK_LEN=chunk_size,
    )
    return dr, dw, dk, dv, da, db, dtau, dh0


@functools.partial(custom_partitioning, static_argnums=(12,))
def _wkv7_sane_bwd_with_mask_spmd(
    r, w, k, v, a, b, sa, state_chkp, tau, mask, dy, dht, chunk_size: int
):
    return _wkv7_sane_bwd_with_mask_triton_call(
        r, w, k, v, a, b, sa, state_chkp, tau, mask, dy, dht, chunk_size
    )


_wkv7_sane_bwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_MASK_RULE,
    partition=_create_partition(_wkv7_sane_bwd_with_mask_triton_call),
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(9,))
def rwkv7_sane_kernel_with_mask_triton(
    r, w, k, v, a, b, tau, mask, h0, chunk_size: int
):
    """带 mask JAX-Triton 训练 kernel 公开入口。"""
    out, sa_out, state_chkp = _wkv7_sane_fwd_with_mask_spmd(
        r, w, k, v, a, b, tau, mask, h0, chunk_size
    )
    final_state = _apply_sane_to_final_state(state_chkp[:, :, -1, :, :], tau, mask=mask)
    return out, final_state


def _fwd_with_mask(r, w, k, v, a, b, tau, mask, h0, chunk_size: int):
    out, sa_out, state_chkp = _wkv7_sane_fwd_with_mask_spmd(
        r, w, k, v, a, b, tau, mask, h0, chunk_size
    )
    final_state = _apply_sane_to_final_state(state_chkp[:, :, -1, :, :], tau, mask=mask)
    return (out, final_state), (r, w, k, v, a, b, tau, mask, sa_out, state_chkp)


def _bwd_with_mask(chunk_size, res, grads):
    r, w, k, v, a, b, tau, mask, sa_out, state_chkp = res
    dy, dht = grads
    dy = jnp.asarray(dy, jnp.bfloat16)
    if dht is None:
        B, N, T, H = r.shape
        dht = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    dr, dw, dk, dv, da, db, dtau, dh0 = _wkv7_sane_bwd_with_mask_spmd(
        r, w, k, v, a, b, sa_out, state_chkp, tau, mask, dy, dht, chunk_size
    )
    return dr, dw, dk, dv, da, db, dtau, None, dh0


rwkv7_sane_kernel_with_mask_triton.defvjp(_fwd_with_mask, _bwd_with_mask)


#  对外 API
def generalized_delta_rule_sane(
    r: jnp.ndarray,
    w: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    tau: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = False,
    chunk_size: int = 16,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """带 State Anomaly Neutralization 的 RWKV-7 广义 delta 规则（JAX-Triton chunkwise 训练版）。

    Args:
        r, w, k, v, a, b: [B, T, H, K], bfloat16。T 必须被 16 整除。
        tau: [B, T//16, H], float32。阈值，必须严格 > 1。
        mask: [B, T//16], float32 或 None。>0 的 chunk 边界执行 SANE；
            仅当 output_final_state=True 时生效。
        initial_state: [B, H, K, K] 或 [1, H, K, K]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先 ([B, H, T, K])。

    Returns:
        out: [B, T, H, K]，bfloat16。
        final_state: [B, H, K, K]，float32。
            output_final_state=False 时不返回；mask=None 时为 None。

    Raises:
        ValueError: T 不被 16 整除，或 tau/mask 形状不匹配。
    """
    dtype = r.dtype

    # 统一转换为 head-first [B, N, T, H]。
    r = _transpose_head(r, head_first)
    w = _transpose_head(w, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    a = _transpose_head(a, head_first)
    b = _transpose_head(b, head_first)
    tau = _transpose_tau(tau)

    B, N, T, H = r.shape
    if T % chunk_size != 0:
        raise ValueError(
            f"Triton SANE kernel requires sequence length T={T} to be divisible by {chunk_size}"
        )
    if chunk_size < 16:
        raise ValueError(f"Triton kernel requires chunk_size >= 16, got {chunk_size}")

    if tau.shape != (B, N, T // chunk_size):
        raise ValueError(
            f"tau shape {tau.shape} does not match expected (B={B}, N={N}, T//16={T // chunk_size})"
        )

    if initial_state is None:
        h0 = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        h0 = jnp.asarray(initial_state, dtype=jnp.float32)

    use_mask = output_final_state and mask is not None

    if use_mask:
        mask = jnp.asarray(mask, dtype=jnp.float32)
        if mask.shape != (B, T // chunk_size):
            raise ValueError(
                f"mask shape {mask.shape} must match (B, T//16) = ({B}, {T // chunk_size})"
            )
        out, last_state = rwkv7_sane_kernel_with_mask_triton(
            r, w, k, v, a, b, tau, mask, h0, chunk_size
        )
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = jnp.asarray(out, dtype)
        return (out, last_state) if output_final_state else out

    # 无 mask 路径：chunk 边界无条件执行 SANE。
    out, _ = rwkv7_sane_kernel_triton(r, w, k, v, a, b, tau, h0, chunk_size)
    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if not output_final_state:
        return out

    warnings.warn(
        "[rwkv7_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
        "由于未提供 padding mask，返回的 final_state 可能被污染，"
        "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
        "[rwkv7_sane] mask is None: using unconditional State Anomaly Neutralization. "
        "The returned final_state is set to None because padding chunks "
        "may contaminate the state. Provide an explicit mask to obtain final_state.",
        UserWarning,
        stacklevel=2,
    )
    return out, None


def generalized_delta_rule_sane_inference(
    r: jnp.ndarray,
    w: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    tau: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Triton 版本推理入口：直接复用训练 kernel，T 仍需被 16 整除。"""
    return generalized_delta_rule_sane(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        tau=tau,
        mask=mask,
        initial_state=initial_state,
        output_final_state=output_final_state,
        head_first=head_first,
    )


def get_jax_generalized_delta_rule_sane(HEAD_SIZE=64, chunk_size: int = 16):
    """返回绑定 chunk_size 的 JAX-Triton 后端 (训练算子, 推理算子)。"""
    import functools

    return [
        functools.partial(generalized_delta_rule_sane, chunk_size=chunk_size),
        functools.partial(generalized_delta_rule_sane_inference, chunk_size=chunk_size),
    ]
