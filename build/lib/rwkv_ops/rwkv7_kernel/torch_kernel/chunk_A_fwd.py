# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch
import triton

from ..triton_kernel.utils import is_gather_supported

from ..triton_kernel.chunk_A_fwd import *


def chunk_dplr_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    scale: float,
    chunk_size: int,
):
    B, T, H, K = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    NT = triton.cdiv(T, BT)

    Aqk = q.new_empty(B, T, H, BT, dtype=q.dtype)
    Aqb = q.new_empty(B, T, H, BT, dtype=q.dtype)
    # involving matrix inverse and it'd be better to use float here.
    Aab = q.new_empty(B, T, H, BT, dtype=torch.float)
    Aak = q.new_empty(B, T, H, BT, dtype=torch.float)

    grid = (NT, B, H)
    BK = triton.next_power_of_2(K)
    qg = torch.empty_like(q)
    kg = torch.empty_like(k, dtype=q.dtype)
    ag = torch.empty_like(a, dtype=q.dtype)
    bg = torch.empty_like(b, dtype=q.dtype)
    chunk_dplr_fwd_A_kernel_intra_sub_intra[grid](
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        Aqk=Aqk,
        Aqb=Aqb,
        Aab=Aab,
        Aak=Aak,
        qg=qg,
        kg=kg,
        ag=ag,
        bg=bg,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BT,
        BK=BK,
        GATHER_SUPPORTED=is_gather_supported,
    )
    return Aab, Aqk, Aak, Aqb, qg, kg, ag, bg
