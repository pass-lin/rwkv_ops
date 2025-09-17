# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025,Qingwen Lin

from typing import Optional, Tuple

import jax_triton as jt
import jax
import triton

from ..get_jax_devices_info import check_shared_mem
from ..triton_kernel.chunk_h_fwd import *


def chunk_dplr_fwd_h(
    kg: jax.Array,
    v: jax.Array,
    w: jax.Array,
    u: jax.Array,
    bg: jax.Array,
    gk: jax.Array,
    initial_state: Optional[jax.Array] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> Tuple[jax.Array, jax.Array]:
    B, T, H, K, V = *kg.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    # H100 can have larger block size

    if check_shared_mem("hopper"):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif check_shared_mem("ampere"):  # A100
        BV = 32
        BC = 32
    else:
        BV = 16
        BC = 16

    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, (
        "NK > 1 is not supported because it involves time-consuming synchronization"
    )

    out_shapes = [
        jax.ShapeDtypeStruct((B, NT, H, K, V), kg.dtype),
        jax.ShapeDtypeStruct([N, H, K, V], "float32"),
        jax.ShapeDtypeStruct(u.shape, u.dtype),
    ]
    grid = (NK, NV, N * H)
    if initial_state is None:
        initial_state = jax.numpy.zeros([N, H, K, V], "float32")
    h, final_state, v_new = jt.triton_call(
        kg,
        v,
        w,
        bg,
        u,
        gk,
        initial_state,
        T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        kernel=chunk_dplr_fwd_kernel_h.fn,
        out_shape=out_shapes,
        grid=grid,
        STORE_FINAL_STATE=True,
        USE_INITIAL_STATE=True,
    )
    return h, v_new, final_state
