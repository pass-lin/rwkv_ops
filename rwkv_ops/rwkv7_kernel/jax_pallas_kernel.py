"""
JAX 版 RWKV-7 Pallas Kernel 封装

设计目标：
- 作为 KERAS_BACKEND=jax 且 KERNEL_TYPE=native（或显式 pallas）时
  非 CPU 平台（GPU/TPU）的默认 kernel。
- kernel 本体只使用稳定的公开 Pallas API（pl.program_id / ref 索引 /
  lax.fori_loop / jnp），不写死任何后端私有 API，
  以便同时兼容 Triton 与 Mosaic GPU 后端，以及 TPU。
- 后端配置、autotune 与 SPMD partition 回调统一由
  rwkv_ops.pallas_utils 提供；custom_partitioning 即使在 eager 调用下
  也会 trace 内层函数，因此每个 custom_vjp 入口（primal/_fwd/_bwd）
  都先用真实数组调用对应 warmup（内部调 ensure_config 解析并缓存配置），
  再走 SPMD 包装（其内层只会被 trace，只查缓存）。
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.custom_partitioning import custom_partitioning

from ..pallas_utils import create_partition, ensure_config, launch

from jax.sharding import NamedSharding, PartitionSpec

CHUNK_LEN = 16


# =========================================================================
# SPMD 切分规则 (Einsum 风格)
# b=Batch, n=Head, t=Time, h=HeadDim1, m=HeadDim2, c=Chunk
# =========================================================================
FWD_RULE = "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n h m -> b n t h, b n t h, b n c h m"
BWD_RULE = "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n c h m, b n h m -> b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n h m"

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


def _transpose_head(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.bfloat16)
    if not head_first:
        # [B, T, N, H] -> [B, N, T, H]
        return jnp.transpose(x, (0, 2, 1, 3))
    return x


# =========================================================================
# Pallas Kernel 本体（纯公开 API，后端无关）
# =========================================================================
def _rwkv7_fwd_kernel(
    r_ref, w_ref, k_ref, v_ref, a_ref, b_ref, h0_ref, o_ref, sa_ref, chkp_ref
):
    b = pl.program_id(0)
    h = pl.program_id(1)
    num_chunks = r_ref.shape[2] // CHUNK_LEN

    state = h0_ref[b, h].astype(jnp.float32)

    def chunk_body(c, state):
        for j in range(CHUNK_LEN):
            t = c * CHUNK_LEN + j
            rv = r_ref[b, h, t, :].astype(jnp.float32)
            wv = w_ref[b, h, t, :].astype(jnp.float32)
            kv = k_ref[b, h, t, :].astype(jnp.float32)
            vv = v_ref[b, h, t, :].astype(jnp.float32)
            av = a_ref[b, h, t, :].astype(jnp.float32)
            bv = b_ref[b, h, t, :].astype(jnp.float32)

            w_decay = jnp.exp(-jnp.exp(wv))
            sa_vec = jnp.sum(state * av[None, :], axis=1)
            sa_ref[b, h, t, :] = sa_vec

            state = (
                state * w_decay[None, :]
                + sa_vec[:, None] * bv[None, :]
                + vv[:, None] * kv[None, :]
            )

            y_vec = jnp.sum(state * rv[None, :], axis=1)
            o_ref[b, h, t, :] = y_vec.astype(o_ref.dtype)

        chkp_ref[b, h, c] = state
        return state

    jax.lax.fori_loop(0, num_chunks, chunk_body, state)


def _rwkv7_fwd_kernel_with_mask(
    r_ref, w_ref, k_ref, v_ref, a_ref, b_ref, mask_ref, h0_ref, o_ref, sa_ref, chkp_ref
):
    b = pl.program_id(0)
    h = pl.program_id(1)
    num_chunks = r_ref.shape[2] // CHUNK_LEN

    state = h0_ref[b, h].astype(jnp.float32)

    def chunk_body(c, state):
        for j in range(CHUNK_LEN):
            t = c * CHUNK_LEN + j
            rv = r_ref[b, h, t, :].astype(jnp.float32)
            wv = w_ref[b, h, t, :].astype(jnp.float32)
            kv = k_ref[b, h, t, :].astype(jnp.float32)
            vv = v_ref[b, h, t, :].astype(jnp.float32)
            av = a_ref[b, h, t, :].astype(jnp.float32)
            bv = b_ref[b, h, t, :].astype(jnp.float32)
            m_val = mask_ref[b, t].astype(jnp.float32)

            w_decay = jnp.exp(-jnp.exp(wv))
            sa_vec = jnp.sum(state * av[None, :], axis=1)
            sa_ref[b, h, t, :] = sa_vec

            state_cand = (
                state * w_decay[None, :]
                + sa_vec[:, None] * bv[None, :]
                + vv[:, None] * kv[None, :]
            )

            y_vec = jnp.sum(state_cand * rv[None, :], axis=1)
            o_ref[b, h, t, :] = y_vec.astype(o_ref.dtype)

            state = m_val * state_cand + (1.0 - m_val) * state

        chkp_ref[b, h, c] = state
        return state

    jax.lax.fori_loop(0, num_chunks, chunk_body, state)


def _rwkv7_bwd_kernel(
    r_ref,
    w_ref,
    k_ref,
    v_ref,
    a_ref,
    b_ref,
    sa_ref,
    chkp_ref,
    dy_ref,
    dht_ref,
    dr_ref,
    dw_ref,
    dk_ref,
    dv_ref,
    da_ref,
    db_ref,
    dh0_ref,
):
    b = pl.program_id(0)
    h = pl.program_id(1)
    num_chunks = r_ref.shape[2] // CHUNK_LEN

    dS = dht_ref[b, h].astype(jnp.float32)

    def chunk_body(c_rev, dS):
        c = num_chunks - 1 - c_rev
        S_t = chkp_ref[b, h, c].astype(jnp.float32)
        for j in range(CHUNK_LEN - 1, -1, -1):
            t = c * CHUNK_LEN + j
            rv = r_ref[b, h, t, :].astype(jnp.float32)
            wv = w_ref[b, h, t, :].astype(jnp.float32)
            kv = k_ref[b, h, t, :].astype(jnp.float32)
            vv = v_ref[b, h, t, :].astype(jnp.float32)
            av = a_ref[b, h, t, :].astype(jnp.float32)
            bv = b_ref[b, h, t, :].astype(jnp.float32)
            dyv = dy_ref[b, h, t, :].astype(jnp.float32)
            sav = sa_ref[b, h, t, :].astype(jnp.float32)

            w_decay = jnp.exp(-jnp.exp(wv))
            w_grad_factor = w_decay * (-jnp.exp(wv))

            dr = jnp.sum(S_t * dyv[:, None], axis=0)
            dr_ref[b, h, t, :] = dr.astype(dr_ref.dtype)

            inv_w = 1.0 / (w_decay + 1e-6)
            S_t = (
                S_t - vv[:, None] * kv[None, :] - sav[:, None] * bv[None, :]
            ) * inv_w[None, :]

            dS = dS + dyv[:, None] * rv[None, :]

            dw = jnp.sum(dS * S_t, axis=0) * w_grad_factor
            dk = jnp.sum(dS * vv[:, None], axis=0)
            dv = jnp.sum(dS * kv[None, :], axis=1)
            db = jnp.sum(dS * sav[:, None], axis=0)
            dsa = jnp.sum(dS * bv[None, :], axis=1)
            da = jnp.sum(S_t * dsa[:, None], axis=0)

            dw_ref[b, h, t, :] = dw.astype(dw_ref.dtype)
            dk_ref[b, h, t, :] = dk.astype(dk_ref.dtype)
            dv_ref[b, h, t, :] = dv.astype(dv_ref.dtype)
            db_ref[b, h, t, :] = db.astype(db_ref.dtype)
            da_ref[b, h, t, :] = da.astype(da_ref.dtype)

            dS = dS * w_decay[None, :] + dsa[:, None] * av[None, :]
        return dS

    dS = jax.lax.fori_loop(0, num_chunks, chunk_body, dS)
    dh0_ref[b, h] = dS.astype(dh0_ref.dtype)


def _rwkv7_bwd_kernel_with_mask(
    r_ref,
    w_ref,
    k_ref,
    v_ref,
    a_ref,
    b_ref,
    mask_ref,
    sa_ref,
    chkp_ref,
    dy_ref,
    dht_ref,
    dr_ref,
    dw_ref,
    dk_ref,
    dv_ref,
    da_ref,
    db_ref,
    dh0_ref,
):
    b = pl.program_id(0)
    h = pl.program_id(1)
    num_chunks = r_ref.shape[2] // CHUNK_LEN

    dS = dht_ref[b, h].astype(jnp.float32)

    def chunk_body(c_rev, dS):
        c = num_chunks - 1 - c_rev
        S_t = chkp_ref[b, h, c].astype(jnp.float32)
        for j in range(CHUNK_LEN - 1, -1, -1):
            t = c * CHUNK_LEN + j
            rv = r_ref[b, h, t, :].astype(jnp.float32)
            wv = w_ref[b, h, t, :].astype(jnp.float32)
            kv = k_ref[b, h, t, :].astype(jnp.float32)
            vv = v_ref[b, h, t, :].astype(jnp.float32)
            av = a_ref[b, h, t, :].astype(jnp.float32)
            bv = b_ref[b, h, t, :].astype(jnp.float32)
            dyv = dy_ref[b, h, t, :].astype(jnp.float32)
            sav = sa_ref[b, h, t, :].astype(jnp.float32)
            m_val = mask_ref[b, t].astype(jnp.float32)

            w_decay = jnp.exp(-jnp.exp(wv))
            w_grad_factor = w_decay * (-jnp.exp(wv))

            s_cand = (
                S_t * w_decay[None, :]
                + sav[:, None] * bv[None, :]
                + vv[:, None] * kv[None, :]
            )
            s_for_dr = m_val * S_t + (1.0 - m_val) * s_cand
            dr = jnp.sum(s_for_dr * dyv[:, None], axis=0)
            dr_ref[b, h, t, :] = dr.astype(dr_ref.dtype)

            dS_curr = dyv[:, None] * rv[None, :]
            dS = dS + dS_curr
            dS_old = dS

            inv_w = 1.0 / (w_decay + 1e-6)
            S_prev = (
                S_t - vv[:, None] * kv[None, :] - sav[:, None] * bv[None, :]
            ) * inv_w[None, :]
            S_t = m_val * S_prev + (1.0 - m_val) * S_t

            dS_param = m_val * dS_old + (1.0 - m_val) * dS_curr

            dw = jnp.sum(dS_param * S_t, axis=0) * w_grad_factor
            dk = jnp.sum(dS_param * vv[:, None], axis=0)
            dv = jnp.sum(dS_param * kv[None, :], axis=1)
            db = jnp.sum(dS_param * sav[:, None], axis=0)
            dsa = jnp.sum(dS_param * bv[None, :], axis=1)
            da = jnp.sum(S_t * dsa[:, None], axis=0)

            dw_ref[b, h, t, :] = dw.astype(dw_ref.dtype)
            dk_ref[b, h, t, :] = dk.astype(dk_ref.dtype)
            dv_ref[b, h, t, :] = dv.astype(dv_ref.dtype)
            db_ref[b, h, t, :] = db.astype(db_ref.dtype)
            da_ref[b, h, t, :] = da.astype(da_ref.dtype)

            trans = dS_param * w_decay[None, :] + dsa[:, None] * av[None, :]
            penetration = dS_old - dS_param
            dS = trans + (1.0 - m_val) * penetration
        return dS

    dS = jax.lax.fori_loop(0, num_chunks, chunk_body, dS)
    dh0_ref[b, h] = dS.astype(dh0_ref.dtype)


# =========================================================================
# 无 Mask 的 Pallas Launcher
# =========================================================================
def _fwd_out_shape(r):
    B, N, T, H = r.shape
    return [
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # OUT
        jax.ShapeDtypeStruct((B, N, T, H), jnp.float32),  # SA_OUT
        jax.ShapeDtypeStruct((B, N, T // CHUNK_LEN, H, H), jnp.float32),  # STATE_CHKP
    ]


def _bwd_out_shape(r):
    B, N, T, H = r.shape
    return [
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # DR
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # DW
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # DK
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # DV
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # DA
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # DB
        jax.ShapeDtypeStruct((B, N, H, H), jnp.float32),  # DH0
    ]


def _wkv7_fwd_pallas_call(r, w, k, v, a, b, h0):
    B, N, T, H = r.shape
    return launch(
        "wkv7_fwd",
        _rwkv7_fwd_kernel,
        _fwd_out_shape(r),
        (B, N),
        (r, w, k, v, a, b, h0),
    )


def _wkv7_fwd_warmup(r, w, k, v, a, b, h0):
    B, N, T, H = r.shape
    ensure_config(
        "wkv7_fwd",
        _rwkv7_fwd_kernel,
        _fwd_out_shape(r),
        (B, N),
        (r, w, k, v, a, b, h0),
    )


def _wkv7_bwd_warmup(r, w, k, v, a, b, sa, state_chkp, dy, dht):
    B, N, T, H = r.shape
    ensure_config(
        "wkv7_bwd",
        _rwkv7_bwd_kernel,
        _bwd_out_shape(r),
        (B, N),
        (r, w, k, v, a, b, sa, state_chkp, dy, dht),
    )


def _wkv7_fwd_with_mask_warmup(r, w, k, v, a, b, h0, mask):
    B, N, T, H = r.shape
    ensure_config(
        "wkv7_fwd_mask",
        _rwkv7_fwd_kernel_with_mask,
        _fwd_out_shape(r),
        (B, N),
        (r, w, k, v, a, b, mask, h0),
    )


def _wkv7_bwd_with_mask_warmup(r, w, k, v, a, b, mask, dy, sa, state_chkp, dht):
    B, N, T, H = r.shape
    ensure_config(
        "wkv7_bwd_mask",
        _rwkv7_bwd_kernel_with_mask,
        _bwd_out_shape(r),
        (B, N),
        (r, w, k, v, a, b, mask, sa, state_chkp, dy, dht),
    )


@custom_partitioning
def _wkv7_fwd_spmd(r, w, k, v, a, b, h0):
    return _wkv7_fwd_pallas_call(r, w, k, v, a, b, h0)


_wkv7_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=create_partition(_wkv7_fwd_pallas_call),
)


def _wkv7_bwd_pallas_call(r, w, k, v, a, b, sa, state_chkp, dy, dht):
    B, N, T, H = r.shape
    return launch(
        "wkv7_bwd",
        _rwkv7_bwd_kernel,
        _bwd_out_shape(r),
        (B, N),
        (r, w, k, v, a, b, sa, state_chkp, dy, dht),
    )


@custom_partitioning
def _wkv7_bwd_spmd(r, w, k, v, a, b, dy, sa, state_chkp, dht):
    return _wkv7_bwd_pallas_call(r, w, k, v, a, b, sa, state_chkp, dy, dht)


_wkv7_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=create_partition(_wkv7_bwd_pallas_call),
)


@jax.custom_vjp
def rwkv7_kernel_pallas(r, w, k, v, a, b, h0):
    _wkv7_fwd_warmup(r, w, k, v, a, b, h0)
    out, sa_out, state_chkp = _wkv7_fwd_spmd(r, w, k, v, a, b, h0)
    final_state = state_chkp[:, :, -1, :, :]
    return out, final_state


def _fwd(r, w, k, v, a, b, h0):
    _wkv7_fwd_warmup(r, w, k, v, a, b, h0)
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

    _wkv7_bwd_warmup(r, w, k, v, a, b, sa_out, state_chkp, dy, dht)
    dr, dw, dk, dv, da, db, dh0 = _wkv7_bwd_spmd(
        r, w, k, v, a, b, dy, sa_out, state_chkp, dht
    )
    return dr, dw, dk, dv, da, db, dh0


rwkv7_kernel_pallas.defvjp(_fwd, _bwd)


# =========================================================================
# 带 Mask 的 Pallas Launcher
# =========================================================================
def _wkv7_fwd_with_mask_pallas_call(r, w, k, v, a, b, h0, mask):
    B, N, T, H = r.shape
    return launch(
        "wkv7_fwd_mask",
        _rwkv7_fwd_kernel_with_mask,
        _fwd_out_shape(r),
        (B, N),
        (r, w, k, v, a, b, mask, h0),
    )


@custom_partitioning
def _wkv7_fwd_with_mask_spmd(r, w, k, v, a, b, h0, mask):
    return _wkv7_fwd_with_mask_pallas_call(r, w, k, v, a, b, h0, mask)


_wkv7_fwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_MASK_RULE,
    partition=create_partition(_wkv7_fwd_with_mask_pallas_call),
)


def _wkv7_bwd_with_mask_pallas_call(r, w, k, v, a, b, mask, dy, sa, state_chkp, dht):
    B, N, T, H = r.shape
    return launch(
        "wkv7_bwd_mask",
        _rwkv7_bwd_kernel_with_mask,
        _bwd_out_shape(r),
        (B, N),
        (r, w, k, v, a, b, mask, sa, state_chkp, dy, dht),
    )


@custom_partitioning
def _wkv7_bwd_with_mask_spmd(r, w, k, v, a, b, mask, dy, sa, state_chkp, dht):
    return _wkv7_bwd_with_mask_pallas_call(
        r, w, k, v, a, b, mask, dy, sa, state_chkp, dht
    )


_wkv7_bwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_MASK_RULE,
    partition=create_partition(_wkv7_bwd_with_mask_pallas_call),
)


@jax.custom_vjp
def rwkv7_kernel_with_mask_pallas(r, w, k, v, a, b, h0, mask):
    _wkv7_fwd_with_mask_warmup(r, w, k, v, a, b, h0, mask)
    out, sa_out, state_chkp = _wkv7_fwd_with_mask_spmd(r, w, k, v, a, b, h0, mask)
    final_state = state_chkp[:, :, -1, :, :]
    return out, final_state


def _fwd_with_mask(r, w, k, v, a, b, h0, mask):
    _wkv7_fwd_with_mask_warmup(r, w, k, v, a, b, h0, mask)
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

    _wkv7_bwd_with_mask_warmup(r, w, k, v, a, b, mask, dy, sa_out, state_chkp, dht)
    dr, dw, dk, dv, da, db, dh0 = _wkv7_bwd_with_mask_spmd(
        r, w, k, v, a, b, mask, dy, sa_out, state_chkp, dht
    )
    return dr, dw, dk, dv, da, db, dh0, None


rwkv7_kernel_with_mask_pallas.defvjp(_fwd_with_mask, _bwd_with_mask)


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
    dtype = r.dtype
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
            f"Pallas kernel requires sequence length T={T} to be divisible by {CHUNK_LEN}"
        )

    # 准备初始状态
    if initial_state is None:
        h0 = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        h0 = jnp.asarray(initial_state, dtype=jnp.float32)

    # 路由调用 Mask 还是 Non-Mask
    if mask is None:
        out, last_state = rwkv7_kernel_pallas(r, w, k, v, a, b, h0)
    else:
        if mask.shape != (B, T) and mask.shape != (B, T, 1, 1):
            raise ValueError(
                f"Mask shape must be (B, T) or (B, T, 1, 1), got {mask.shape}"
            )
        mask = jnp.asarray(mask, dtype=jnp.float32).reshape(B, T)
        out, last_state = rwkv7_kernel_with_mask_pallas(r, w, k, v, a, b, h0, mask)

    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if output_final_state:
        return out, last_state
    return out


def get_jax_generalized_delta_rule(HEAD_SIZE=64):
    return generalized_delta_rule, generalized_delta_rule
