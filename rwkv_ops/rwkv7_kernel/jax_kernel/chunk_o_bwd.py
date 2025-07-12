# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Tuple

import jax_triton as jt
import jax
import triton

from ..get_jax_devices_info import check_shared_mem
from ..triton_kernel.chunk_o_bwd import *


def chunk_dplr_bwd_dv(
    A_qk: jax.Array,
    kg: jax.Array,
    do: jax.Array,
    dh: jax.Array,
    chunk_size: int = 64,
) -> jax.Array:
    B, T, H, K, V = *kg.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    NT = triton.cdiv(T, BT)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    dv = jt.triton_call(
        A_qk,
        kg,
        do,
        dh,
        T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        kernel=chunk_dplr_bwd_kernel_dv,
        out_shape=jax.ShapeDtypeStruct(do.shape, do.dtype),
        grid=grid,
    )
    return dv


def chunk_dplr_bwd_o(
    k: jax.Array,
    b: jax.Array,
    v: jax.Array,
    v_new: jax.Array,
    gk: jax.Array,
    do: jax.Array,
    h: jax.Array,
    dh: jax.Array,
    dv: jax.Array,
    w: jax.Array,
    chunk_size: int = 64,
    scale: float = 1.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    B, T, H, K, V = *w.shape, v.shape[-1]

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)

    BK = (
        min(triton.next_power_of_2(K), 64)
        if check_shared_mem()
        else min(triton.next_power_of_2(K), 32)
    )
    BV = (
        min(triton.next_power_of_2(V), 64)
        if check_shared_mem()
        else min(triton.next_power_of_2(K), 32)
    )
    NK = triton.cdiv(K, BK)
    grid = (NK, NT, B * H)

    out_shapes = [
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(w.shape, w.dtype),
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct([B, NT, H, K], "float32"),
    ]
    dq, dk, dw, db, dgk_last = jt.triton_call(
        v,
        v_new,
        h,
        do,
        dh,
        w,
        dv,
        gk,
        k,
        b,
        T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        kernel=chunk_dplr_bwd_o_kernel,
        out_shape=out_shapes,
        grid=grid,
    )
    return (dq, dk, dw, db, dgk_last)


def chunk_dplr_bwd_dAu(
    v: jax.Array,
    v_new: jax.Array,
    do: jax.Array,
    A_qb: jax.Array,
    scale: float,
    chunk_size: int = 64,
) -> jax.Array:
    B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)

    if check_shared_mem("ampere"):  # A100
        BV = min(triton.next_power_of_2(V), 128)
    elif check_shared_mem("ada"):  # 4090
        BV = min(triton.next_power_of_2(V), 64)
    else:
        BV = min(triton.next_power_of_2(V), 32)

    grid = (NT, B * H)
    out_shapes = [
        jax.ShapeDtypeStruct([B, T, H, BT], "float32"),
        jax.ShapeDtypeStruct([B, T, H, BT], "float32"),
        jax.ShapeDtypeStruct(v_new.shape, v_new.dtype),
    ]
    dA_qk, dA_qb, dv_new = jt.triton_call(
        v,
        do,
        v_new,
        A_qb,
        T,
        scale=scale,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
        grid=grid,
        out_shape=out_shapes,
        kernel=chunk_dplr_bwd_kernel_dAu,
    )
    return dv_new, dA_qk, dA_qb
