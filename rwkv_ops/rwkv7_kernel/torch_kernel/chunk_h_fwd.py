# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton

from ..get_torch_devices_info import check_shared_mem
from ..triton_kernel.chunk_h_fwd import *


def chunk_dplr_fwd_h(
    kg: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    bg: torch.Tensor,
    gk: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *kg.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    # H100 can have larger block size

    if check_shared_mem("hopper", kg.device.index):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif check_shared_mem("ampere", kg.device.index):  # A100
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

    h = kg.new_empty(B, NT, H, K, V)
    final_state = (
        kg.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )
    v_new = torch.empty_like(u)
    grid = (NK, NV, N * H)
    chunk_dplr_fwd_kernel_h[grid](
        kg=kg,
        v=v,
        w=w,
        bg=bg,
        u=u,
        v_new=v_new,
        h=h,
        gk=gk,
        h0=initial_state,
        ht=final_state,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
    )
    return h, v_new, final_state
