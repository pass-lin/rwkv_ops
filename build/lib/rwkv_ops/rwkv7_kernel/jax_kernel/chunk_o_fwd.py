# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025,Qingwen Lin


import jax_triton as jt
import jax
import triton

from ..triton_kernel.chunk_o_fwd import *


def chunk_dplr_fwd_o(
    qg: jax.Array,
    v: jax.Array,
    v_new: jax.Array,
    A_qk: jax.Array,
    A_qb: jax.Array,
    h: jax.Array,
    chunk_size: int = 64,
) -> jax.Array:
    B, T, H, K, V = *qg.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    NT = triton.cdiv(T, BT)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    o = jt.triton_call(
        qg,
        v,
        v_new,
        A_qk,
        A_qb,
        h,
        T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        kernel=chunk_dplr_fwd_kernel_o,
        out_shape=jax.ShapeDtypeStruct(v.shape, v.dtype),
        grid=grid,
    )
    return o
