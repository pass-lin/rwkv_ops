# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025,Qingwen Lin


import jax_triton as jt
import jax
import triton

from ..triton_kernel.utils import is_gather_supported

from ..triton_kernel.chunk_A_fwd import *


def chunk_dplr_fwd_intra(
    q: jax.Array,
    k: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gi: jax.Array,
    ge: jax.Array,
    scale: float,
    chunk_size: int,
):
    B, T, H, K = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    NT = triton.cdiv(T, BT)
    shape = [B, T, H, BT]
    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, q.dtype),
        jax.ShapeDtypeStruct(a.shape, q.dtype),
        jax.ShapeDtypeStruct(b.shape, q.dtype),
        jax.ShapeDtypeStruct(shape, q.dtype),
        jax.ShapeDtypeStruct(shape, q.dtype),
        jax.ShapeDtypeStruct(shape, "float32"),
        jax.ShapeDtypeStruct(shape, "float32"),
    ]
    grid = (NT, B, H)
    BK = triton.next_power_of_2(K)
    qg, kg, ag, bg, Aqk, Aqb, Aab, Aak = jt.triton_call(
        q,
        k,
        a,
        b,
        gi,
        ge,
        T,
        scale=scale,
        H=H,
        K=K,
        BT=BT,
        BC=BT,
        BK=BK,
        GATHER_SUPPORTED=is_gather_supported,
        kernel=chunk_dplr_fwd_A_kernel_intra_sub_intra,
        out_shape=out_shapes,
        grid=grid,
    )
    return Aab, Aqk, Aak, Aqb, qg, kg, ag, bg
