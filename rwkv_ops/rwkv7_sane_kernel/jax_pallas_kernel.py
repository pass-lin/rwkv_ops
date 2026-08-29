"""JAX 版 RWKV7-SANE Pallas kernel 封装。"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

from ..pallas_utils import create_partition, ensure_config, launch


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


def _transpose_head(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.bfloat16)
    if not head_first:
        # [B, T, N, H] -> [B, N, T, H]
        return jnp.transpose(x, (0, 2, 1, 3))
    return x


def _transpose_tau(tau: jnp.ndarray) -> jnp.ndarray:
    """tau 公共接口始终为 [B, T//chunk_size, N]；需要转成 head-first [B, N, T//chunk_size]。"""
    tau = jnp.asarray(tau, dtype=jnp.float32)
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


def _make_rwkv7_sane_fwd_kernel(chunk_size: int):
    def _rwkv7_sane_fwd_kernel(
        r_ref,
        w_ref,
        k_ref,
        v_ref,
        a_ref,
        b_ref,
        tau_ref,
        h0_ref,
        o_ref,
        sa_ref,
        chkp_ref,
    ):
        b = pl.program_id(0)
        h = pl.program_id(1)
        num_chunks = r_ref.shape[2] // chunk_size

        state = h0_ref[b, h].astype(jnp.float32)

        def chunk_body(c, state):
            for j in range(chunk_size):
                t = c * chunk_size + j
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

            # checkpoint 保存 SANE 之前的 state 供反向使用。
            chkp_ref[b, h, c] = state
            tau_v = tau_ref[b, h, c].astype(jnp.float32)
            tau_safe = jnp.maximum(tau_v, 1e-6)
            state = tau_safe * jnp.tanh(state / tau_safe)
            return state

        jax.lax.fori_loop(0, num_chunks, chunk_body, state)

    return _rwkv7_sane_fwd_kernel


def _make_rwkv7_sane_fwd_kernel_with_mask(chunk_size: int):
    def _rwkv7_sane_fwd_kernel_with_mask(
        r_ref,
        w_ref,
        k_ref,
        v_ref,
        a_ref,
        b_ref,
        tau_ref,
        mask_ref,
        h0_ref,
        o_ref,
        sa_ref,
        chkp_ref,
    ):
        b = pl.program_id(0)
        h = pl.program_id(1)
        num_chunks = r_ref.shape[2] // chunk_size

        state = h0_ref[b, h].astype(jnp.float32)

        def chunk_body(c, state):
            for j in range(chunk_size):
                t = c * chunk_size + j
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

            # checkpoint 保存 SANE 之前的 state 供反向使用。
            chkp_ref[b, h, c] = state
            tau_v = tau_ref[b, h, c].astype(jnp.float32)
            tau_safe = jnp.maximum(tau_v, 1e-6)
            sane_state = tau_safe * jnp.tanh(state / tau_safe)
            m = mask_ref[b, c].astype(jnp.float32)
            state = state * (1.0 - m) + sane_state * m
            return state

        jax.lax.fori_loop(0, num_chunks, chunk_body, state)

    return _rwkv7_sane_fwd_kernel_with_mask


def _make_rwkv7_sane_bwd_kernel(chunk_size: int):
    def _rwkv7_sane_bwd_kernel(
        r_ref,
        w_ref,
        k_ref,
        v_ref,
        a_ref,
        b_ref,
        sa_ref,
        chkp_ref,
        tau_ref,
        dy_ref,
        dht_ref,
        dr_ref,
        dw_ref,
        dk_ref,
        dv_ref,
        da_ref,
        db_ref,
        dtau_ref,
        dh0_ref,
    ):
        b = pl.program_id(0)
        h = pl.program_id(1)
        num_chunks = r_ref.shape[2] // chunk_size

        dS = dht_ref[b, h].astype(jnp.float32)

        def chunk_body(c_rev, dS):
            c = num_chunks - 1 - c_rev
            # chkp 保存的是 SANE 之前的 state。
            S_t = chkp_ref[b, h, c].astype(jnp.float32)

            # 先对下游梯度 dS 应用 SANE 导数。
            tau_v = tau_ref[b, h, c].astype(jnp.float32)
            tau_safe = jnp.maximum(tau_v, 1e-6)
            u = S_t / tau_safe
            tnh = jnp.tanh(u)
            sech2 = 1.0 - tnh * tnh
            dtau_local = jnp.sum(dS * (tnh - u * sech2))
            dS = dS * sech2
            dtau_ref[b, h, c] = dtau_local

            for j in range(chunk_size - 1, -1, -1):
                t = c * chunk_size + j
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

    return _rwkv7_sane_bwd_kernel


def _make_rwkv7_sane_bwd_kernel_with_mask(chunk_size: int):
    def _rwkv7_sane_bwd_kernel_with_mask(
        r_ref,
        w_ref,
        k_ref,
        v_ref,
        a_ref,
        b_ref,
        sa_ref,
        chkp_ref,
        tau_ref,
        mask_ref,
        dy_ref,
        dht_ref,
        dr_ref,
        dw_ref,
        dk_ref,
        dv_ref,
        da_ref,
        db_ref,
        dtau_ref,
        dh0_ref,
    ):
        b = pl.program_id(0)
        h = pl.program_id(1)
        num_chunks = r_ref.shape[2] // chunk_size

        dS = dht_ref[b, h].astype(jnp.float32)

        def chunk_body(c_rev, dS):
            c = num_chunks - 1 - c_rev
            # chkp 保存的是 SANE 之前的 state。
            S_t = chkp_ref[b, h, c].astype(jnp.float32)

            # 先对下游梯度 dS 应用 SANE 导数（按 mask blend）。
            tau_v = tau_ref[b, h, c].astype(jnp.float32)
            tau_safe = jnp.maximum(tau_v, 1e-6)
            m = mask_ref[b, c].astype(jnp.float32)
            u = S_t / tau_safe
            tnh = jnp.tanh(u)
            sech2 = 1.0 - tnh * tnh
            blend = (1.0 - m) + m * sech2
            dtau_local = jnp.sum(dS * m * (tnh - u * sech2))
            dS = dS * blend
            dtau_ref[b, h, c] = dtau_local

            for j in range(chunk_size - 1, -1, -1):
                t = c * chunk_size + j
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

    return _rwkv7_sane_bwd_kernel_with_mask


def _fwd_out_shape(r, chunk_size: int):
    B, N, T, H = r.shape
    return [
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # OUT
        jax.ShapeDtypeStruct((B, N, T, H), jnp.float32),  # SA_OUT
        jax.ShapeDtypeStruct((B, N, T // chunk_size, H, H), jnp.float32),  # STATE_CHKP
    ]


def _bwd_out_shape(r, chunk_size: int):
    B, N, T, H = r.shape
    return [
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # DR
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # DW
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # DK
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # DV
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # DA
        jax.ShapeDtypeStruct((B, N, T, H), r.dtype),  # DB
        jax.ShapeDtypeStruct((B, N, T // chunk_size), jnp.float32),  # DTAU
        jax.ShapeDtypeStruct((B, N, H, H), jnp.float32),  # DH0
    ]


def _wkv7_sane_fwd_pallas_call(r, w, k, v, a, b, tau, h0, chunk_size: int):
    B, N, T, H = r.shape
    return launch(
        f"wkv7_sane_fwd_{chunk_size}",
        _make_rwkv7_sane_fwd_kernel(chunk_size),
        _fwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, tau, h0),
    )


def _wkv7_sane_fwd_warmup(r, w, k, v, a, b, tau, h0, chunk_size: int):
    B, N, T, H = r.shape
    ensure_config(
        f"wkv7_sane_fwd_{chunk_size}",
        _make_rwkv7_sane_fwd_kernel(chunk_size),
        _fwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, tau, h0),
    )


def _wkv7_sane_bwd_warmup(
    r, w, k, v, a, b, sa, state_chkp, tau, dy, dht, chunk_size: int
):
    B, N, T, H = r.shape
    ensure_config(
        f"wkv7_sane_bwd_{chunk_size}",
        _make_rwkv7_sane_bwd_kernel(chunk_size),
        _bwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, sa, state_chkp, tau, dy, dht),
    )


@custom_partitioning
def _wkv7_sane_fwd_spmd(r, w, k, v, a, b, tau, h0, chunk_size: int):
    return _wkv7_sane_fwd_pallas_call(r, w, k, v, a, b, tau, h0, chunk_size)


_wkv7_sane_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=create_partition(_wkv7_sane_fwd_pallas_call),
)


def _wkv7_sane_bwd_pallas_call(
    r, w, k, v, a, b, sa, state_chkp, tau, dy, dht, chunk_size: int
):
    B, N, T, H = r.shape
    return launch(
        f"wkv7_sane_bwd_{chunk_size}",
        _make_rwkv7_sane_bwd_kernel(chunk_size),
        _bwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, sa, state_chkp, tau, dy, dht),
    )


@custom_partitioning
def _wkv7_sane_bwd_spmd(
    r, w, k, v, a, b, sa, state_chkp, tau, dy, dht, chunk_size: int
):
    return _wkv7_sane_bwd_pallas_call(
        r, w, k, v, a, b, sa, state_chkp, tau, dy, dht, chunk_size
    )


_wkv7_sane_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=create_partition(_wkv7_sane_bwd_pallas_call),
)


@jax.custom_vjp
def rwkv7_sane_kernel_pallas(r, w, k, v, a, b, tau, h0, chunk_size: int):
    """无 mask Pallas 训练 kernel 公开入口。"""
    _wkv7_sane_fwd_warmup(r, w, k, v, a, b, tau, h0, chunk_size)
    out, sa_out, state_chkp = _wkv7_sane_fwd_spmd(r, w, k, v, a, b, tau, h0, chunk_size)
    final_state = _apply_sane_to_final_state(state_chkp[:, :, -1, :, :], tau)
    return out, final_state


def _fwd(r, w, k, v, a, b, tau, h0, chunk_size: int):
    _wkv7_sane_fwd_warmup(r, w, k, v, a, b, tau, h0, chunk_size)
    out, sa_out, state_chkp = _wkv7_sane_fwd_spmd(r, w, k, v, a, b, tau, h0, chunk_size)
    final_state = _apply_sane_to_final_state(state_chkp[:, :, -1, :, :], tau)
    return (out, final_state), (
        r,
        w,
        k,
        v,
        a,
        b,
        tau,
        sa_out,
        state_chkp,
        chunk_size,
    )


def _bwd(res, grads):
    r, w, k, v, a, b, tau, sa_out, state_chkp, chunk_size = res
    dy, dht = grads
    dy = jnp.asarray(dy, jnp.bfloat16)
    if dht is None:
        B, N, T, H = r.shape
        dht = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    _wkv7_sane_bwd_warmup(
        r, w, k, v, a, b, sa_out, state_chkp, tau, dy, dht, chunk_size
    )
    dr, dw, dk, dv, da, db, dtau, dh0 = _wkv7_sane_bwd_spmd(
        r, w, k, v, a, b, sa_out, state_chkp, tau, dy, dht, chunk_size
    )
    return dr, dw, dk, dv, da, db, dtau, dh0


rwkv7_sane_kernel_pallas.defvjp(_fwd, _bwd)


#  带 mask launcher
def _wkv7_sane_fwd_with_mask_pallas_call(
    r, w, k, v, a, b, tau, mask, h0, chunk_size: int
):
    B, N, T, H = r.shape
    return launch(
        f"wkv7_sane_fwd_mask_{chunk_size}",
        _make_rwkv7_sane_fwd_kernel_with_mask(chunk_size),
        _fwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, tau, mask, h0),
    )


def _wkv7_sane_fwd_with_mask_warmup(r, w, k, v, a, b, tau, mask, h0, chunk_size: int):
    B, N, T, H = r.shape
    ensure_config(
        f"wkv7_sane_fwd_mask_{chunk_size}",
        _make_rwkv7_sane_fwd_kernel_with_mask(chunk_size),
        _fwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, tau, mask, h0),
    )


def _wkv7_sane_bwd_with_mask_warmup(
    r, w, k, v, a, b, sa, state_chkp, tau, mask, dy, dht, chunk_size: int
):
    B, N, T, H = r.shape
    ensure_config(
        f"wkv7_sane_bwd_mask_{chunk_size}",
        _make_rwkv7_sane_bwd_kernel_with_mask(chunk_size),
        _bwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, sa, state_chkp, tau, mask, dy, dht),
    )


@custom_partitioning
def _wkv7_sane_fwd_with_mask_spmd(r, w, k, v, a, b, tau, mask, h0, chunk_size: int):
    return _wkv7_sane_fwd_with_mask_pallas_call(
        r, w, k, v, a, b, tau, mask, h0, chunk_size
    )


_wkv7_sane_fwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_MASK_RULE,
    partition=create_partition(_wkv7_sane_fwd_with_mask_pallas_call),
)


def _wkv7_sane_bwd_with_mask_pallas_call(
    r, w, k, v, a, b, sa, state_chkp, tau, mask, dy, dht, chunk_size: int
):
    B, N, T, H = r.shape
    return launch(
        f"wkv7_sane_bwd_mask_{chunk_size}",
        _make_rwkv7_sane_bwd_kernel_with_mask(chunk_size),
        _bwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, sa, state_chkp, tau, mask, dy, dht),
    )


@custom_partitioning
def _wkv7_sane_bwd_with_mask_spmd(
    r, w, k, v, a, b, sa, state_chkp, tau, mask, dy, dht, chunk_size: int
):
    return _wkv7_sane_bwd_with_mask_pallas_call(
        r, w, k, v, a, b, sa, state_chkp, tau, mask, dy, dht, chunk_size
    )


_wkv7_sane_bwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_MASK_RULE,
    partition=create_partition(_wkv7_sane_bwd_with_mask_pallas_call),
)


@jax.custom_vjp
def rwkv7_sane_kernel_with_mask_pallas(
    r, w, k, v, a, b, tau, mask, h0, chunk_size: int
):
    """带 mask Pallas 训练 kernel 公开入口。"""
    _wkv7_sane_fwd_with_mask_warmup(r, w, k, v, a, b, tau, mask, h0, chunk_size)
    out, sa_out, state_chkp = _wkv7_sane_fwd_with_mask_spmd(
        r, w, k, v, a, b, tau, mask, h0, chunk_size
    )
    final_state = _apply_sane_to_final_state(state_chkp[:, :, -1, :, :], tau, mask=mask)
    return out, final_state


def _fwd_with_mask(r, w, k, v, a, b, tau, mask, h0, chunk_size: int):
    _wkv7_sane_fwd_with_mask_warmup(r, w, k, v, a, b, tau, mask, h0, chunk_size)
    out, sa_out, state_chkp = _wkv7_sane_fwd_with_mask_spmd(
        r, w, k, v, a, b, tau, mask, h0, chunk_size
    )
    final_state = _apply_sane_to_final_state(state_chkp[:, :, -1, :, :], tau, mask=mask)
    return (out, final_state), (
        r,
        w,
        k,
        v,
        a,
        b,
        tau,
        mask,
        sa_out,
        state_chkp,
        chunk_size,
    )


def _bwd_with_mask(res, grads):
    r, w, k, v, a, b, tau, mask, sa_out, state_chkp, chunk_size = res
    dy, dht = grads
    dy = jnp.asarray(dy, jnp.bfloat16)
    if dht is None:
        B, N, T, H = r.shape
        dht = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    _wkv7_sane_bwd_with_mask_warmup(
        r, w, k, v, a, b, sa_out, state_chkp, tau, mask, dy, dht, chunk_size
    )
    dr, dw, dk, dv, da, db, dtau, dh0 = _wkv7_sane_bwd_with_mask_spmd(
        r, w, k, v, a, b, sa_out, state_chkp, tau, mask, dy, dht, chunk_size
    )
    return dr, dw, dk, dv, da, db, dtau, None, dh0


rwkv7_sane_kernel_with_mask_pallas.defvjp(_fwd_with_mask, _bwd_with_mask)


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
    """带 State Anomaly Neutralization 的 RWKV-7 广义 delta 规则（Pallas chunkwise 训练版）。

    Args:
        r, w, k, v, a, b: [B, T, H, K]，bfloat16。T 必须被 chunk_size 整除。
        tau: [B, T//chunk_size, H], float32。阈值，必须严格 > 1。
        mask: [B, T//chunk_size], float32 或 None。>0 的 chunk 边界执行 SANE；
            仅当 output_final_state=True 时生效。
        initial_state: [B, H, K, K] 或 [1, H, K, K]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, K]）。
        chunk_size: int，chunk 长度，必须整除序列长度。

    Returns:
        out: [B, T, H, K]，bfloat16。
        final_state: [B, H, K, K]，float32。
            output_final_state=False 时不返回；mask=None 时为 None。

    Raises:
        ValueError: T 不被 chunk_size 整除，或 tau/mask 形状不匹配。
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
            f"Pallas SANE kernel requires sequence length T={T} to be divisible by {chunk_size}"
        )

    if tau.shape != (B, N, T // chunk_size):
        raise ValueError(
            f"tau shape {tau.shape} does not match expected "
            f"(B={B}, N={N}, T//chunk_size={T // chunk_size})"
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
                f"mask shape {mask.shape} must match (B, T//chunk_size) = ({B}, {T // chunk_size})"
            )
        out, last_state = rwkv7_sane_kernel_with_mask_pallas(
            r, w, k, v, a, b, tau, mask, h0, chunk_size
        )
        out = jnp.asarray(out, dtype)
        return out, last_state

    out, last_state = rwkv7_sane_kernel_pallas(r, w, k, v, a, b, tau, h0, chunk_size)
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


def get_jax_generalized_delta_rule_sane(HEAD_SIZE=64, chunk_size: int = 16):
    """返回 RWKV-7-SANE Pallas 训练/推理算子对（当前两者相同）。"""
    return generalized_delta_rule_sane, generalized_delta_rule_sane
