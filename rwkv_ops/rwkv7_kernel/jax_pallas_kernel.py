"""JAX 版 RWKV7 Pallas kernel 封装。"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.custom_partitioning import custom_partitioning

from ..pallas_utils import create_partition, ensure_config, launch

from jax.sharding import NamedSharding, PartitionSpec


#  SPMD 切分规则（Einsum 风格）
# b=Batch, n=Head, t=Time, h=HeadDim1, m=HeadDim2, c=Chunk
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


def _make_rwkv7_fwd_kernel(chunk_size: int):
    def _rwkv7_fwd_kernel(
        r_ref, w_ref, k_ref, v_ref, a_ref, b_ref, h0_ref, o_ref, sa_ref, chkp_ref
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

            chkp_ref[b, h, c] = state
            return state

        jax.lax.fori_loop(0, num_chunks, chunk_body, state)

    return _rwkv7_fwd_kernel


def _make_rwkv7_fwd_kernel_with_mask(chunk_size: int):
    def _rwkv7_fwd_kernel_with_mask(
        r_ref,
        w_ref,
        k_ref,
        v_ref,
        a_ref,
        b_ref,
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

    return _rwkv7_fwd_kernel_with_mask


def _make_rwkv7_bwd_kernel(chunk_size: int):
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
        num_chunks = r_ref.shape[2] // chunk_size

        dS = dht_ref[b, h].astype(jnp.float32)

        def chunk_body(c_rev, dS):
            c = num_chunks - 1 - c_rev
            S_t = chkp_ref[b, h, c].astype(jnp.float32)
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

    return _rwkv7_bwd_kernel


def _make_rwkv7_bwd_kernel_with_mask(chunk_size: int):
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
        num_chunks = r_ref.shape[2] // chunk_size

        dS = dht_ref[b, h].astype(jnp.float32)

        def chunk_body(c_rev, dS):
            c = num_chunks - 1 - c_rev
            S_t = chkp_ref[b, h, c].astype(jnp.float32)
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

    return _rwkv7_bwd_kernel_with_mask


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
        jax.ShapeDtypeStruct((B, N, H, H), jnp.float32),  # DH0
    ]


def _wkv7_fwd_pallas_call(r, w, k, v, a, b, h0, chunk_size: int):
    B, N, T, H = r.shape
    return launch(
        f"wkv7_fwd_{chunk_size}",
        _make_rwkv7_fwd_kernel(chunk_size),
        _fwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, h0),
    )


def _wkv7_fwd_warmup(r, w, k, v, a, b, h0, chunk_size: int):
    B, N, T, H = r.shape
    ensure_config(
        f"wkv7_fwd_{chunk_size}",
        _make_rwkv7_fwd_kernel(chunk_size),
        _fwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, h0),
    )


def _wkv7_bwd_warmup(r, w, k, v, a, b, sa, state_chkp, dy, dht, chunk_size: int):
    B, N, T, H = r.shape
    ensure_config(
        f"wkv7_bwd_{chunk_size}",
        _make_rwkv7_bwd_kernel(chunk_size),
        _bwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, sa, state_chkp, dy, dht),
    )


def _wkv7_fwd_with_mask_warmup(r, w, k, v, a, b, h0, mask, chunk_size: int):
    B, N, T, H = r.shape
    ensure_config(
        f"wkv7_fwd_mask_{chunk_size}",
        _make_rwkv7_fwd_kernel_with_mask(chunk_size),
        _fwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, mask, h0),
    )


def _wkv7_bwd_with_mask_warmup(
    r, w, k, v, a, b, mask, dy, sa, state_chkp, dht, chunk_size: int
):
    B, N, T, H = r.shape
    ensure_config(
        f"wkv7_bwd_mask_{chunk_size}",
        _make_rwkv7_bwd_kernel_with_mask(chunk_size),
        _bwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, mask, sa, state_chkp, dy, dht),
    )


@custom_partitioning
def _wkv7_fwd_spmd(r, w, k, v, a, b, h0, chunk_size: int):
    return _wkv7_fwd_pallas_call(r, w, k, v, a, b, h0, chunk_size)


_wkv7_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=create_partition(_wkv7_fwd_pallas_call),
)


def _wkv7_bwd_pallas_call(r, w, k, v, a, b, sa, state_chkp, dy, dht, chunk_size: int):
    B, N, T, H = r.shape
    return launch(
        f"wkv7_bwd_{chunk_size}",
        _make_rwkv7_bwd_kernel(chunk_size),
        _bwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, sa, state_chkp, dy, dht),
    )


@custom_partitioning
def _wkv7_bwd_spmd(r, w, k, v, a, b, dy, sa, state_chkp, dht, chunk_size: int):
    return _wkv7_bwd_pallas_call(r, w, k, v, a, b, sa, state_chkp, dy, dht, chunk_size)


_wkv7_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=create_partition(_wkv7_bwd_pallas_call),
)


@jax.custom_vjp
def rwkv7_kernel_pallas(r, w, k, v, a, b, h0, chunk_size: int):
    _wkv7_fwd_warmup(r, w, k, v, a, b, h0, chunk_size)
    out, sa_out, state_chkp = _wkv7_fwd_spmd(r, w, k, v, a, b, h0, chunk_size)
    final_state = state_chkp[:, :, -1, :, :]
    return out, final_state


def _fwd(r, w, k, v, a, b, h0, chunk_size: int):
    _wkv7_fwd_warmup(r, w, k, v, a, b, h0, chunk_size)
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

    _wkv7_bwd_warmup(r, w, k, v, a, b, sa_out, state_chkp, dy, dht, chunk_size)
    dr, dw, dk, dv, da, db, dh0 = _wkv7_bwd_spmd(
        r, w, k, v, a, b, dy, sa_out, state_chkp, dht, chunk_size
    )
    return dr, dw, dk, dv, da, db, dh0


rwkv7_kernel_pallas.defvjp(_fwd, _bwd)


#  带 Mask 的 Pallas Launcher
def _wkv7_fwd_with_mask_pallas_call(r, w, k, v, a, b, h0, mask, chunk_size: int):
    B, N, T, H = r.shape
    return launch(
        f"wkv7_fwd_mask_{chunk_size}",
        _make_rwkv7_fwd_kernel_with_mask(chunk_size),
        _fwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, mask, h0),
    )


@custom_partitioning
def _wkv7_fwd_with_mask_spmd(r, w, k, v, a, b, h0, mask, chunk_size: int):
    return _wkv7_fwd_with_mask_pallas_call(r, w, k, v, a, b, h0, mask, chunk_size)


_wkv7_fwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_MASK_RULE,
    partition=create_partition(_wkv7_fwd_with_mask_pallas_call),
)


def _wkv7_bwd_with_mask_pallas_call(
    r, w, k, v, a, b, mask, dy, sa, state_chkp, dht, chunk_size: int
):
    B, N, T, H = r.shape
    return launch(
        f"wkv7_bwd_mask_{chunk_size}",
        _make_rwkv7_bwd_kernel_with_mask(chunk_size),
        _bwd_out_shape(r, chunk_size),
        (B, N),
        (r, w, k, v, a, b, mask, sa, state_chkp, dy, dht),
    )


@custom_partitioning
def _wkv7_bwd_with_mask_spmd(
    r, w, k, v, a, b, mask, dy, sa, state_chkp, dht, chunk_size: int
):
    return _wkv7_bwd_with_mask_pallas_call(
        r, w, k, v, a, b, mask, dy, sa, state_chkp, dht, chunk_size
    )


_wkv7_bwd_with_mask_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_MASK_RULE,
    partition=create_partition(_wkv7_bwd_with_mask_pallas_call),
)


@jax.custom_vjp
def rwkv7_kernel_with_mask_pallas(r, w, k, v, a, b, h0, mask, chunk_size: int):
    _wkv7_fwd_with_mask_warmup(r, w, k, v, a, b, h0, mask, chunk_size)
    out, sa_out, state_chkp = _wkv7_fwd_with_mask_spmd(
        r, w, k, v, a, b, h0, mask, chunk_size
    )
    final_state = state_chkp[:, :, -1, :, :]
    return out, final_state


def _fwd_with_mask(r, w, k, v, a, b, h0, mask, chunk_size: int):
    _wkv7_fwd_with_mask_warmup(r, w, k, v, a, b, h0, mask, chunk_size)
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

    _wkv7_bwd_with_mask_warmup(
        r, w, k, v, a, b, mask, dy, sa_out, state_chkp, dht, chunk_size
    )
    dr, dw, dk, dv, da, db, dh0 = _wkv7_bwd_with_mask_spmd(
        r, w, k, v, a, b, mask, dy, sa_out, state_chkp, dht, chunk_size
    )
    return dr, dw, dk, dv, da, db, dh0, None


rwkv7_kernel_with_mask_pallas.defvjp(_fwd_with_mask, _bwd_with_mask)


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
    """RWKV-7 chunkwise 训练算子（JAX Pallas 实现）。

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
            f"Pallas kernel requires sequence length T={T} to be divisible by {chunk_size}"
        )

    # 准备初始状态
    if initial_state is None:
        h0 = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        h0 = jnp.asarray(initial_state, dtype=jnp.float32)

    # 路由调用 Mask 还是 Non-Mask
    if mask is None:
        out, last_state = rwkv7_kernel_pallas(r, w, k, v, a, b, h0, chunk_size)
    else:
        if mask.shape != (B, T) and mask.shape != (B, T, 1, 1):
            raise ValueError(
                f"Mask shape must be (B, T) or (B, T, 1, 1), got {mask.shape}"
            )
        mask = jnp.asarray(mask, dtype=jnp.float32).reshape(B, T)
        out, last_state = rwkv7_kernel_with_mask_pallas(
            r, w, k, v, a, b, h0, mask, chunk_size
        )

    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if output_final_state:
        return out, last_state
    return out


def get_jax_generalized_delta_rule(HEAD_SIZE=64, chunk_size: int = 16):
    """返回 RWKV-7 Pallas 训练/推理算子对（当前两者相同）。"""
    return generalized_delta_rule, generalized_delta_rule
