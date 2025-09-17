# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import triton
import triton.language as tl

from ..triton_kernel.utils import exp, use_cuda_graph


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BK", "BV"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_dplr_fwd_kernel_h(
    kg,
    v,
    w,
    bg,
    u,
    gk,
    h0,
    T,
    h,
    ht,
    v_new,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if False:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    o_k = i_k * BK + tl.arange(0, BK)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0 + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(
            h + ((boh + i_t) * H + i_h) * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        b_hc = tl.zeros([BK, BV], dtype=tl.float32)
        # since we need to make all DK in the SRAM. we face serve SRAM memory burden. By subchunking we allievate such burden
        for i_c in range(tl.cdiv(min(BT, T - i_t * BT), BC)):
            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT + i_c * BC),
                (BK, BC),
                (0, 1),
            )
            p_bg = tl.make_block_ptr(
                bg + (bos * H + i_h) * K,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT + i_c * BC),
                (BK, BC),
                (0, 1),
            )
            p_w = tl.make_block_ptr(
                w + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT + i_c * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_v = tl.make_block_ptr(
                v + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT + i_c * BC, i_v * BV),
                (BC, BV),
                (1, 0),
            )
            p_u = tl.make_block_ptr(
                u + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT + i_c * BC, i_v * BV),
                (BC, BV),
                (1, 0),
            )
            p_v_new = tl.make_block_ptr(
                v_new + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT + i_c * BC, i_v * BV),
                (BC, BV),
                (1, 0),
            )
            # [BK, BC]
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_bg = tl.load(p_bg, boundary_check=(0, 1))
            b_v2 = tl.dot(b_w, b_h.to(b_w.dtype)) + tl.load(p_u, boundary_check=(0, 1))
            b_hc += tl.dot(b_kg, b_v)
            b_hc += tl.dot(b_bg.to(b_hc.dtype), b_v2)
            tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(
            gk + (bos + last_idx) * H * K + i_h * K + o_k, mask=o_k < K
        ).to(tl.float32)
        b_h *= exp(b_g_last[:, None])
        b_h += b_hc

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        tl.store(
            p_ht,
            b_h.to(p_ht.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
