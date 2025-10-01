# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025,Qingwen Lin

from typing import Tuple

import jax_triton as jt
import jax
import triton


from ..triton_kernel.wy_fast_fwd import *


def wu_fwd(
    ag: jax.Array,
    v: jax.Array,
    A_ak: jax.Array,
    A_ab_inv: jax.Array,
    chunk_size: int,
) -> Tuple[jax.Array, jax.Array]:
    B, T, H, K, V = *ag.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)

    out_shapes = [
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct(ag.shape, ag.dtype),
    ]
    grid = (NT, B * H)
    w, u = jt.triton_call(
        ag,
        v,
        A_ab_inv,
        A_ak,
        T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        grid=grid,
        kernel=wu_fwd_kernel,
        out_shape=out_shapes,
    )
    return w, u


def prepare_wy_repr_fwd(
    ag: jax.Array,
    v: jax.Array,
    A_ak: jax.Array,
    A_ab: jax.Array,
    chunk_size: int = 64,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    B, T, H, _ = ag.shape
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    NT = triton.cdiv(T, BT)
    BC = min(BT, 32)
    fwd_fn = (
        prepare_wy_repr_fwd_kernel_chunk64
        if BT == 64
        else prepare_wy_repr_fwd_kernel_chunk32
    )
    grid = (NT, B * H)
    A_ab_inv = jt.triton_call(
        A_ab,
        T,
        H=H,
        BT=BT,
        BC=BC,
        grid=grid,
        kernel=fwd_fn,
        out_shape=jax.ShapeDtypeStruct(A_ab.shape, A_ab.dtype),
    )
    w, u = wu_fwd(ag=ag, v=v, A_ak=A_ak, A_ab_inv=A_ab_inv, chunk_size=BT)
    return w, u, A_ab_inv


fwd_prepare_wy_repr = prepare_wy_repr_fwd

fwd_wu = wu_fwd
