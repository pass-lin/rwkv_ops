# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025,Qingwen Lin

from typing import Tuple

import jax_triton as jt
import jax
import triton


from ..get_torch_devices_info import check_shared_mem
from ..triton_kernel.wy_fast_bwd import *


def chunk_dplr_bwd_wy(
    A_ab_inv: jax.Array,
    A_ak: jax.Array,
    v: jax.Array,
    ag: jax.Array,
    dw: jax.Array,
    du: jax.Array,
    dv0: jax.Array,
    chunk_size: int = 16,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    B, T, H, K, V = *dw.shape, du.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = (
        min(triton.next_power_of_2(V), 64)
        if check_shared_mem()
        else min(triton.next_power_of_2(V), 32)
    )
    grid = (NT, B * H)
    out_shapes = [
        jax.ShapeDtypeStruct(A_ak.shape, "float32"),
        jax.ShapeDtypeStruct(A_ab_inv.shape, "float32"),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct(ag.shape, ag.dtype),
    ]
    dA_ak, dA_ab, dv, dag = jt.triton_call(
        A_ab_inv,
        A_ak,
        ag,
        v,
        dw,
        du,
        dv0,
        T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        grid=grid,
        kernel=prepare_wy_repr_bwd_kernel,
        out_shape=out_shapes,
    )
    return dA_ab, dA_ak, dv, dag
