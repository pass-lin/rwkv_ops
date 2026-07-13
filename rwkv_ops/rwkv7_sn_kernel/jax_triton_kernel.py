"""
JAX 版 RWKV-7 State Neutralization Triton Kernel 封装

参考 rwkv7_kernel/jax_triton_kernel.py 的结构，增加：
- tau: per-head per-chunk 阈值，形状 [B, N_HEAD, T//16]
- mask: chunk-level 标志，形状 [B, T//16]
- 无 mask / 带 mask 两套 forward / backward kernel
- 支持 SPMD 切分
"""

from __future__ import annotations
import warnings

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Optional, Tuple, Union

import jax_triton as jt

from .triton_kernel import (
    rwkv7_sn_fwd_kernel,
    rwkv7_sn_bwd_kernel,
    rwkv7_sn_fwd_kernel_with_mask,
    rwkv7_sn_bwd_kernel_with_mask,
)

from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

CHUNK_LEN = 16

# =========================================================================
# SPMD 切分规则 (Einsum 风格)
# b=Batch, n=Head, t=Time, h=HeadDim, c=Chunk
# =========================================================================
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


def _apply_sn_to_final_state(
    state: jnp.ndarray,
    tau: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """state: [B, N, H, H]; tau: [B, N, C]; mask: [B, C]（可选）。"""
    last_tau = tau[:, :, -1][:, :, None, None]
    tau_safe = jnp.maximum(last_tau, 1e-6)
    sn_state = last_tau * jnp.tanh(state / tau_safe)
    if mask is None:
        return sn_state
    last_mask = mask[:, -1][:, None, None, None]
    return jnp.where(last_mask > 0, sn_state, state)


# =========================================================================
# 无 Mask 的 JAX-Triton Launcher
# =========================================================================
def _wkv7_sn_fwd_triton_call(r, w, k, v, a, b, tau, h0):
    B, N, T, H = r.shape
    dtype = r.dtype
    chunk_num = T // CHUNK_LEN

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
        kernel=rwkv7_sn_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,
        CHUNK_LEN=CHUNK_LEN,
    )
    return out, sa_out, state_chkp


@custom_partitioning
def _wkv7_sn_fwd_spmd(r, w, k, v, a, b, tau, h0):
    return _wkv7_sn_fwd_triton_call(r, w, k, v, a, b, tau, h0)


_wkv7_sn_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=_create_partition(_wkv7_sn_fwd_triton_call),
)


def _wkv7_sn_bwd_triton_call(r, w, k, v, a, b, sa, state_chkp, tau, dy, dht):
    B, N, T, H = r.shape
    dtype = r.dtype

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DR
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DW
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DK
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DV
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DA
        jax.ShapeDtypeStruct((B, N, T, H), dtype),  # DB
        jax.ShapeDtypeStruct((B, N, T // CHUNK_LEN), jnp.float32),  # DTAU
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
        kernel=rwkv7_sn_bwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,
        CHUNK_LEN=CHUNK_LEN,
    )
    return dr, dw, dk, dv, da, db, dtau, dh0


@custom_partitioning
def _wkv7_sn_bwd_spmd(r, w, k, v, a, b, sa, state_chkp, tau, dy, dht):
    return _wkv7_sn_bwd_triton_call(r, w, k, v, a, b, sa, state_chkp, tau, dy, dht)


_wkv7_sn_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=_create_partition(_wkv7_sn_bwd_triton_call),
)


@jax.custom_vjp
def rwkv7_sn_kernel_triton(r, w, k, v, a, b, tau, h0):
    out, sa_out, state_chkp = _wkv7_sn_fwd_spmd(r, w, k, v, a, b, tau, h0)
    final_state = _apply_sn_to_final_state(state_chkp[:, :, -1, :, :], tau)
    return out, final_state


def _fwd(r, w, k, v, a, b, tau, h0):
    out, sa_out, state_chkp = _wkv7_sn_fwd_spmd(r, w, k, v, a, b, tau, h0)
    final_state = _apply_sn_to_final_state(state_chkp[:, :, -1, :, :], tau)
    return (out, final_state), (r, w, k, v, a, b, tau, sa_out, state_chkp)


def _bwd(res, grads):
    r, w, k, v, a, b, tau, sa_out, state_chkp = res
    dy, dht = grads
    dy = jnp.asarray(dy, jnp.bfloat16)
    if dht is None:
        B, N, T, H = r.shape
        dht = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    dr, dw, dk, dv, da, db, dtau, dh0 = _wkv7_sn_bwd_spmd(
        r, w, k, v, a, b, sa_out, state_chkp, tau, dy, dht
    )
    return dr, dw, dk, dv, da, db, dtau, dh0


rwkv7_sn_kernel_triton.defvjp(_fwd, _bwd)


# =========================================================================
# 带 Mask 的 JAX-Triton Launcher
# =========================================================================
def _wkv7_sn_fwd_with_mask_triton_call(r, w, k, v, a, b, tau, mask, h0):
    B, N, T, H = r.shape
    dtype = r.dtype
    chunk_num = T // CHUNK_LEN

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
        kernel=rwkv7_sn_fwd_kernel_with_mask,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,
        CHUNK_LEN=CHUNK_LEN,
    )
    return out, sa_out, state_chkp


@custom_partitioning
def _wkv7_sn_fwd_with_mask_spmd(r, w, k, v, a, b, tau, mask, h0):
    return _wkv7_sn_fwd_with_mask_triton_call(r, w, k, v, a, b, tau, mask, h0)


_wkv7_sn_fwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_MASK_RULE,
    partition=_create_partition(_wkv7_sn_fwd_with_mask_triton_call),
)


def _wkv7_sn_bwd_with_mask_triton_call(
    r, w, k, v, a, b, sa, state_chkp, tau, mask, dy, dht
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
        jax.ShapeDtypeStruct((B, N, T // CHUNK_LEN), jnp.float32),  # DTAU
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
        kernel=rwkv7_sn_bwd_kernel_with_mask,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,
        CHUNK_LEN=CHUNK_LEN,
    )
    return dr, dw, dk, dv, da, db, dtau, dh0


@custom_partitioning
def _wkv7_sn_bwd_with_mask_spmd(r, w, k, v, a, b, sa, state_chkp, tau, mask, dy, dht):
    return _wkv7_sn_bwd_with_mask_triton_call(
        r, w, k, v, a, b, sa, state_chkp, tau, mask, dy, dht
    )


_wkv7_sn_bwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_MASK_RULE,
    partition=_create_partition(_wkv7_sn_bwd_with_mask_triton_call),
)


@jax.custom_vjp
def rwkv7_sn_kernel_with_mask_triton(r, w, k, v, a, b, tau, mask, h0):
    out, sa_out, state_chkp = _wkv7_sn_fwd_with_mask_spmd(
        r, w, k, v, a, b, tau, mask, h0
    )
    final_state = _apply_sn_to_final_state(state_chkp[:, :, -1, :, :], tau, mask=mask)
    return out, final_state


def _fwd_with_mask(r, w, k, v, a, b, tau, mask, h0):
    out, sa_out, state_chkp = _wkv7_sn_fwd_with_mask_spmd(
        r, w, k, v, a, b, tau, mask, h0
    )
    final_state = _apply_sn_to_final_state(state_chkp[:, :, -1, :, :], tau, mask=mask)
    return (out, final_state), (r, w, k, v, a, b, tau, mask, sa_out, state_chkp)


def _bwd_with_mask(res, grads):
    r, w, k, v, a, b, tau, mask, sa_out, state_chkp = res
    dy, dht = grads
    dy = jnp.asarray(dy, jnp.bfloat16)
    if dht is None:
        B, N, T, H = r.shape
        dht = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    dr, dw, dk, dv, da, db, dtau, dh0 = _wkv7_sn_bwd_with_mask_spmd(
        r, w, k, v, a, b, sa_out, state_chkp, tau, mask, dy, dht
    )
    return dr, dw, dk, dv, da, db, dtau, None, dh0


rwkv7_sn_kernel_with_mask_triton.defvjp(_fwd_with_mask, _bwd_with_mask)


# =========================================================================
# 对外 API 暴露
# =========================================================================
def generalized_delta_rule_sn(
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
    dtype = r.dtype

    # 统一转换为 head-first [B, N, T, H]
    r = _transpose_head(r, head_first)
    w = _transpose_head(w, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    a = _transpose_head(a, head_first)
    b = _transpose_head(b, head_first)
    tau = _transpose_tau(tau)

    B, N, T, H = r.shape
    if T % CHUNK_LEN != 0:
        raise ValueError(
            f"Triton SN kernel requires sequence length T={T} to be divisible by {CHUNK_LEN}"
        )

    if tau.shape != (B, N, T // CHUNK_LEN):
        raise ValueError(
            f"tau shape {tau.shape} does not match expected (B={B}, N={N}, T//16={T // CHUNK_LEN})"
        )

    if initial_state is None:
        h0 = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        h0 = jnp.asarray(initial_state, dtype=jnp.float32)

    use_mask = output_final_state and mask is not None

    if use_mask:
        mask = jnp.asarray(mask, dtype=jnp.float32)
        if mask.shape != (B, T // CHUNK_LEN):
            raise ValueError(
                f"mask shape {mask.shape} must match (B, T//16) = ({B}, {T // CHUNK_LEN})"
            )
        out, last_state = rwkv7_sn_kernel_with_mask_triton(
            r, w, k, v, a, b, tau, mask, h0
        )
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = jnp.asarray(out, dtype)
        return (out, last_state) if output_final_state else out

    # 无 mask 路径：chunk 边界无条件执行 SN
    out, _ = rwkv7_sn_kernel_triton(r, w, k, v, a, b, tau, h0)
    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if not output_final_state:
        return out

    warnings.warn(
        "[rwkv7_sn] mask is None: 使用无条件 State Neutralization 算子。"
        "由于未提供 padding mask，返回的 final_state 可能被污染，"
        "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
        "[rwkv7_sn] mask is None: using unconditional State Neutralization. "
        "The returned final_state is set to None because padding chunks "
        "may contaminate the state. Provide an explicit mask to obtain final_state.",
        UserWarning,
        stacklevel=2,
    )
    return out, None


def generalized_delta_rule_sn_inference(
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
    return generalized_delta_rule_sn(
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


def get_jax_generalized_delta_rule_sn(HEAD_SIZE=64):
    return [generalized_delta_rule_sn, generalized_delta_rule_sn_inference]
