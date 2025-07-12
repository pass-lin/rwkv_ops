# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton

from ..get_torch_devices_info import check_shared_mem
from ..triton_kernel.chunk_h_bwd import *


def chunk_dplr_bwd_dhu(
    qg: torch.Tensor,
    bg: torch.Tensor,
    w: torch.Tensor,
    gk: torch.Tensor,
    h0: torch.Tensor,
    dht: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *qg.shape, do.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    BK = triton.next_power_of_2(K)
    assert BK <= 256, (
        "current kernel does not support head dimension being larger than 256."
    )
    # H100
    if check_shared_mem("hopper", qg.device.index):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif check_shared_mem("ampere", qg.device.index):  # A100
        BV = 32
        BC = 32
    else:  # Etc: 4090
        BV = 16
        BC = 16

    N, NT, chunk_offsets = B, triton.cdiv(T, BT), None

    BC = min(BT, BC)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, (
        "NK > 1 is not supported because it involves time-consuming synchronization"
    )

    dh = qg.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.zeros_like(dv)

    grid = (NK, NV, N * H)
    chunk_dplr_bwd_kernel_dhu[grid](
        qg=qg,
        bg=bg,
        w=w,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
    )
    return dh, dh0, dv2
