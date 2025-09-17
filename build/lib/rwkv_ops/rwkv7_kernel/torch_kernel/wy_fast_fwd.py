# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Tuple

import torch
import triton

from ..triton_kernel.wy_fast_fwd import *


def wu_fwd(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab_inv: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *ag.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)

    w = torch.empty_like(ag)
    u = torch.empty_like(v)
    wu_fwd_kernel[(NT, B * H)](
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        w=w,
        u=u,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u


def prepare_wy_repr_fwd(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab: torch.Tensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, _ = ag.shape
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    NT = triton.cdiv(T, BT)
    BC = min(BT, 32)
    fwd_fn = (
        prepare_wy_repr_fwd_kernel_chunk64
        if BT == 64
        else prepare_wy_repr_fwd_kernel_chunk32
    )

    A_ab_inv = torch.empty_like(A_ab)
    fwd_fn[(NT, B * H)](
        A_ab=A_ab,
        A_ab_inv=A_ab_inv,
        T=T,
        H=H,
        BT=BT,
        BC=BC,
    )
    w, u = wu_fwd(ag=ag, v=v, A_ak=A_ak, A_ab_inv=A_ab_inv, chunk_size=BT)
    return w, u, A_ab_inv


fwd_prepare_wy_repr = prepare_wy_repr_fwd

fwd_wu = wu_fwd
