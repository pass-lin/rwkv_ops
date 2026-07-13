import triton
import triton.language as tl


# ======================================================================================
#                          无 Mask 前向传播 (chunk 边界无条件 SN)
# ======================================================================================
@triton.autotune(
    configs=[
        triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def rwkv7_sn_fwd_kernel(
    R,
    W,
    K,
    V,
    A,
    B_param,
    TAU,
    H0,
    B_BATCH,
    N_HEAD,
    T_LEN,
    OUT,
    SA_OUT,
    STATE_CHKP,
    H_SIZE: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    MINI_BSZ: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
    b_mask = b_range < B_BATCH
    cols = tl.arange(0, H_SIZE)

    b_range_i64 = b_range.to(tl.int64)
    N_HEAD_i64 = tl.cast(N_HEAD, tl.int64)
    T_LEN_i64 = tl.cast(T_LEN, tl.int64)
    H_SIZE_i64 = tl.cast(H_SIZE, tl.int64)
    pid_h_i64 = tl.cast(pid_h, tl.int64)
    CHUNK_LEN_i64 = tl.cast(CHUNK_LEN, tl.int64)

    ptr_mask = b_mask[:, None]
    state_mask = b_mask[:, None, None]

    base_ptr_off = (
        (b_range_i64[:, None] * N_HEAD_i64 * T_LEN_i64 * H_SIZE_i64)
        + (pid_h_i64 * T_LEN_i64 * H_SIZE_i64)
        + cols[None, :]
    )
    h0_base = (b_range_i64[:, None, None] * N_HEAD_i64 * H_SIZE_i64 * H_SIZE_i64) + (
        pid_h_i64 * H_SIZE_i64 * H_SIZE_i64
    )

    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]
    state = tl.load(
        H0 + h0_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)

    chunk_num_i64 = T_LEN_i64 // CHUNK_LEN_i64
    tau_base_batch = (
        b_range_i64 * N_HEAD_i64 * chunk_num_i64 + pid_h_i64 * chunk_num_i64
    )

    for t in range(0, T_LEN):
        t_off = t * H_SIZE

        rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )

        w_decay = tl.exp(-tl.exp(wv))
        sa_vec = tl.sum(state * av[:, None, :], axis=2)
        tl.store(
            SA_OUT + base_ptr_off + t_off,
            sa_vec.to(SA_OUT.dtype.element_ty),
            mask=ptr_mask,
        )

        state = (
            state * w_decay[:, None, :]
            + sa_vec[:, :, None] * bv[:, None, :]
            + vv[:, :, None] * kv[:, None, :]
        )

        y_vec = tl.sum(state * rv[:, None, :], axis=2)
        tl.store(
            OUT + base_ptr_off + t_off, y_vec.to(OUT.dtype.element_ty), mask=ptr_mask
        )

        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (tl.cast(t, tl.int64) + 1) // CHUNK_LEN_i64 - 1
            chkp_base = (
                (
                    b_range_i64[:, None, None]
                    * N_HEAD_i64
                    * chunk_num_i64
                    * H_SIZE_i64
                    * H_SIZE_i64
                )
                + (pid_h_i64 * chunk_num_i64 * H_SIZE_i64 * H_SIZE_i64)
                + (chkp_t * H_SIZE_i64 * H_SIZE_i64)
            )
            tl.store(
                STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                state,
                mask=state_mask,
            )

            # 在 chunk 边界执行 State Neutralization（无条件）
            tau_v = tl.load(TAU + tau_base_batch + chkp_t, mask=b_mask, other=1.0).to(
                tl.float32
            )
            tau_safe = tl.maximum(tau_v, 1e-6)
            x2 = 2.0 * state / tau_safe[:, None, None]
            tnh = tl.where(
                state >= 0.0,
                1.0 - 2.0 / (tl.exp(x2) + 1.0),
                2.0 / (tl.exp(-x2) + 1.0) - 1.0,
            )
            state = tau_safe[:, None, None] * tnh


# ======================================================================================
#                          无 Mask 反向传播
# ======================================================================================
@triton.autotune(
    configs=[
        triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def rwkv7_sn_bwd_kernel(
    R,
    W,
    K,
    V,
    A,
    B_param,
    SA,
    STATE_CHKP,
    TAU,
    B_BATCH,
    N_HEAD,
    T_LEN,
    DY,
    DHT,
    DR,
    DW,
    DK,
    DV,
    DA,
    DB,
    DTAU,
    DH0,
    H_SIZE: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    MINI_BSZ: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
    b_mask = b_range < B_BATCH
    cols = tl.arange(0, H_SIZE)
    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]

    b_range_i64 = b_range.to(tl.int64)
    N_HEAD_i64 = tl.cast(N_HEAD, tl.int64)
    T_LEN_i64 = tl.cast(T_LEN, tl.int64)
    H_SIZE_i64 = tl.cast(H_SIZE, tl.int64)
    pid_h_i64 = tl.cast(pid_h, tl.int64)
    CHUNK_LEN_i64 = tl.cast(CHUNK_LEN, tl.int64)

    ptr_mask = b_mask[:, None]
    state_mask = b_mask[:, None, None]

    base_ptr_off = (
        (b_range_i64[:, None] * N_HEAD_i64 * T_LEN_i64 * H_SIZE_i64)
        + (pid_h_i64 * T_LEN_i64 * H_SIZE_i64)
        + cols[None, :]
    )
    dht_base = (b_range_i64[:, None, None] * N_HEAD_i64 * H_SIZE_i64 * H_SIZE_i64) + (
        pid_h_i64 * H_SIZE_i64 * H_SIZE_i64
    )

    dS = tl.load(
        DHT + dht_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)
    S_t = tl.zeros([MINI_BSZ, H_SIZE, H_SIZE], dtype=tl.float32)

    chunk_num_i64 = T_LEN_i64 // CHUNK_LEN_i64
    tau_base_batch = (
        b_range_i64 * N_HEAD_i64 * chunk_num_i64 + pid_h_i64 * chunk_num_i64
    )

    for t in range(T_LEN - 1, -1, -1):
        t_off = t * H_SIZE

        rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )
        dyv = tl.load(DY + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )
        sav = tl.load(SA + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )

        w_decay = tl.exp(-tl.exp(wv))
        w_grad_factor = w_decay * (-tl.exp(wv))

        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (tl.cast(t, tl.int64) + 1) // CHUNK_LEN_i64 - 1
            chkp_base = (
                (
                    b_range_i64[:, None, None]
                    * N_HEAD_i64
                    * chunk_num_i64
                    * H_SIZE_i64
                    * H_SIZE_i64
                )
                + (pid_h_i64 * chunk_num_i64 * H_SIZE_i64 * H_SIZE_i64)
                + (chkp_t * H_SIZE_i64 * H_SIZE_i64)
            )
            S_t = tl.load(
                STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                mask=state_mask,
                other=0.0,
            ).to(tl.float32)

            # State Neutralization 反向：先对下游梯度 dS 应用 SN 导数
            tau_v = tl.load(TAU + tau_base_batch + chkp_t, mask=b_mask, other=1.0).to(
                tl.float32
            )
            tau_safe = tl.maximum(tau_v, 1e-6)
            u = S_t / tau_safe[:, None, None]
            x2 = 2.0 * u
            tnh = tl.where(
                u >= 0.0,
                1.0 - 2.0 / (tl.exp(x2) + 1.0),
                2.0 / (tl.exp(-x2) + 1.0) - 1.0,
            )
            sech2 = 1.0 - tnh * tnh

            dtau_local = tl.sum(tl.sum(dS * (tnh - u * sech2), axis=2), axis=1)
            tl.store(DTAU + tau_base_batch + chkp_t, dtau_local, mask=b_mask)
            dS = dS * sech2

        dr = tl.sum(S_t * dyv[:, :, None], axis=1)
        tl.store(DR + base_ptr_off + t_off, dr.to(DR.dtype.element_ty), mask=ptr_mask)

        inv_w = 1.0 / (w_decay + 1e-6)
        S_t = (
            S_t - vv[:, :, None] * kv[:, None, :] - sav[:, :, None] * bv[:, None, :]
        ) * inv_w[:, None, :]

        dS = dS + dyv[:, :, None] * rv[:, None, :]

        dw = tl.sum(dS * S_t, axis=1) * w_grad_factor
        dk = tl.sum(dS * vv[:, :, None], axis=1)
        dv = tl.sum(dS * kv[:, None, :], axis=2)
        db = tl.sum(dS * sav[:, :, None], axis=1)
        dsa = tl.sum(dS * bv[:, None, :], axis=2)
        da = tl.sum(S_t * dsa[:, :, None], axis=1)

        tl.store(DW + base_ptr_off + t_off, dw.to(DW.dtype.element_ty), mask=ptr_mask)
        tl.store(DK + base_ptr_off + t_off, dk.to(DK.dtype.element_ty), mask=ptr_mask)
        tl.store(DV + base_ptr_off + t_off, dv.to(DV.dtype.element_ty), mask=ptr_mask)
        tl.store(DB + base_ptr_off + t_off, db.to(DB.dtype.element_ty), mask=ptr_mask)
        tl.store(DA + base_ptr_off + t_off, da.to(DA.dtype.element_ty), mask=ptr_mask)

        dS = dS * w_decay[:, None, :] + dsa[:, :, None] * av[:, None, :]

        if t == 0:
            tl.store(
                DH0 + dht_base + row_idx * H_SIZE + col_idx,
                dS.to(DH0.dtype.element_ty),
                mask=state_mask,
            )


# ======================================================================================
#                          带 Mask 前向传播 (chunk 边界按 mask 选择 SN)
# ======================================================================================
@triton.autotune(
    configs=[
        triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def rwkv7_sn_fwd_kernel_with_mask(
    R,
    W,
    K,
    V,
    A,
    B_param,
    TAU,
    MASK,
    H0,
    B_BATCH,
    N_HEAD,
    T_LEN,
    OUT,
    SA_OUT,
    STATE_CHKP,
    H_SIZE: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    MINI_BSZ: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
    b_mask = b_range < B_BATCH
    cols = tl.arange(0, H_SIZE)

    b_range_i64 = b_range.to(tl.int64)
    N_HEAD_i64 = tl.cast(N_HEAD, tl.int64)
    T_LEN_i64 = tl.cast(T_LEN, tl.int64)
    H_SIZE_i64 = tl.cast(H_SIZE, tl.int64)
    pid_h_i64 = tl.cast(pid_h, tl.int64)
    CHUNK_LEN_i64 = tl.cast(CHUNK_LEN, tl.int64)

    ptr_mask = b_mask[:, None]
    state_mask = b_mask[:, None, None]

    base_ptr_off = (
        (b_range_i64[:, None] * N_HEAD_i64 * T_LEN_i64 * H_SIZE_i64)
        + (pid_h_i64 * T_LEN_i64 * H_SIZE_i64)
        + cols[None, :]
    )
    h0_base = (b_range_i64[:, None, None] * N_HEAD_i64 * H_SIZE_i64 * H_SIZE_i64) + (
        pid_h_i64 * H_SIZE_i64 * H_SIZE_i64
    )
    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]

    state = tl.load(
        H0 + h0_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)

    chunk_num_i64 = T_LEN_i64 // CHUNK_LEN_i64
    tau_base_batch = (
        b_range_i64 * N_HEAD_i64 * chunk_num_i64 + pid_h_i64 * chunk_num_i64
    )
    mask_base_batch = b_range_i64 * chunk_num_i64

    for t in range(0, T_LEN):
        t_off = t * H_SIZE

        rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )

        w_decay = tl.exp(-tl.exp(wv))
        sa_vec = tl.sum(state * av[:, None, :], axis=2)
        tl.store(
            SA_OUT + base_ptr_off + t_off,
            sa_vec.to(SA_OUT.dtype.element_ty),
            mask=ptr_mask,
        )

        state = (
            state * w_decay[:, None, :]
            + sa_vec[:, :, None] * bv[:, None, :]
            + vv[:, :, None] * kv[:, None, :]
        )

        y_vec = tl.sum(state * rv[:, None, :], axis=2)
        tl.store(
            OUT + base_ptr_off + t_off, y_vec.to(OUT.dtype.element_ty), mask=ptr_mask
        )

        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (tl.cast(t, tl.int64) + 1) // CHUNK_LEN_i64 - 1
            chkp_base = (
                (
                    b_range_i64[:, None, None]
                    * N_HEAD_i64
                    * chunk_num_i64
                    * H_SIZE_i64
                    * H_SIZE_i64
                )
                + (pid_h_i64 * chunk_num_i64 * H_SIZE_i64 * H_SIZE_i64)
                + (chkp_t * H_SIZE_i64 * H_SIZE_i64)
            )
            tl.store(
                STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                state,
                mask=state_mask,
            )

            # 按 chunk-level mask 选择是否执行 State Neutralization
            tau_v = tl.load(TAU + tau_base_batch + chkp_t, mask=b_mask, other=1.0).to(
                tl.float32
            )
            tau_safe = tl.maximum(tau_v, 1e-6)
            m_val = tl.load(MASK + mask_base_batch + chkp_t, mask=b_mask, other=0.0).to(
                tl.float32
            )
            x2 = 2.0 * state / tau_safe[:, None, None]
            sn_state = tau_safe[:, None, None] * tl.where(
                state >= 0.0,
                1.0 - 2.0 / (tl.exp(x2) + 1.0),
                2.0 / (tl.exp(-x2) + 1.0) - 1.0,
            )
            state = (
                state * (1.0 - m_val[:, None, None]) + sn_state * m_val[:, None, None]
            )


# ======================================================================================
#                          带 Mask 反向传播
# ======================================================================================
@triton.autotune(
    configs=[
        triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def rwkv7_sn_bwd_kernel_with_mask(
    R,
    W,
    K,
    V,
    A,
    B_param,
    SA,
    STATE_CHKP,
    TAU,
    MASK,
    B_BATCH,
    N_HEAD,
    T_LEN,
    DY,
    DHT,
    DR,
    DW,
    DK,
    DV,
    DA,
    DB,
    DTAU,
    DH0,
    H_SIZE: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    MINI_BSZ: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
    b_mask = b_range < B_BATCH
    cols = tl.arange(0, H_SIZE)
    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]

    b_range_i64 = b_range.to(tl.int64)
    N_HEAD_i64 = tl.cast(N_HEAD, tl.int64)
    T_LEN_i64 = tl.cast(T_LEN, tl.int64)
    H_SIZE_i64 = tl.cast(H_SIZE, tl.int64)
    pid_h_i64 = tl.cast(pid_h, tl.int64)
    CHUNK_LEN_i64 = tl.cast(CHUNK_LEN, tl.int64)

    ptr_mask = b_mask[:, None]
    state_mask = b_mask[:, None, None]

    base_ptr_off = (
        (b_range_i64[:, None] * N_HEAD_i64 * T_LEN_i64 * H_SIZE_i64)
        + (pid_h_i64 * T_LEN_i64 * H_SIZE_i64)
        + cols[None, :]
    )
    dht_base = (b_range_i64[:, None, None] * N_HEAD_i64 * H_SIZE_i64 * H_SIZE_i64) + (
        pid_h_i64 * H_SIZE_i64 * H_SIZE_i64
    )

    dS = tl.load(
        DHT + dht_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)
    S_t = tl.zeros([MINI_BSZ, H_SIZE, H_SIZE], dtype=tl.float32)

    chunk_num_i64 = T_LEN_i64 // CHUNK_LEN_i64
    tau_base_batch = (
        b_range_i64 * N_HEAD_i64 * chunk_num_i64 + pid_h_i64 * chunk_num_i64
    )
    mask_base_batch = b_range_i64 * chunk_num_i64

    for t in range(T_LEN - 1, -1, -1):
        t_off = t * H_SIZE

        rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )
        dyv = tl.load(DY + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )
        sav = tl.load(SA + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )

        w_decay = tl.exp(-tl.exp(wv))
        w_grad_factor = w_decay * (-tl.exp(wv))

        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (tl.cast(t, tl.int64) + 1) // CHUNK_LEN_i64 - 1
            chkp_base = (
                (
                    b_range_i64[:, None, None]
                    * N_HEAD_i64
                    * chunk_num_i64
                    * H_SIZE_i64
                    * H_SIZE_i64
                )
                + (pid_h_i64 * chunk_num_i64 * H_SIZE_i64 * H_SIZE_i64)
                + (chkp_t * H_SIZE_i64 * H_SIZE_i64)
            )
            S_t = tl.load(
                STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                mask=state_mask,
                other=0.0,
            ).to(tl.float32)

            tau_v = tl.load(TAU + tau_base_batch + chkp_t, mask=b_mask, other=1.0).to(
                tl.float32
            )
            tau_safe = tl.maximum(tau_v, 1e-6)
            m_val = tl.load(MASK + mask_base_batch + chkp_t, mask=b_mask, other=0.0).to(
                tl.float32
            )

            u = S_t / tau_safe[:, None, None]
            x2 = 2.0 * u
            tnh = tl.where(
                u >= 0.0,
                1.0 - 2.0 / (tl.exp(x2) + 1.0),
                2.0 / (tl.exp(-x2) + 1.0) - 1.0,
            )
            sech2 = 1.0 - tnh * tnh
            blend = (1.0 - m_val) + m_val * sech2

            dtau_local = tl.sum(
                tl.sum(dS * m_val[:, None, None] * (tnh - u * sech2), axis=2),
                axis=1,
            )
            tl.store(DTAU + tau_base_batch + chkp_t, dtau_local, mask=b_mask)
            dS = dS * blend

        dr = tl.sum(S_t * dyv[:, :, None], axis=1)
        tl.store(DR + base_ptr_off + t_off, dr.to(DR.dtype.element_ty), mask=ptr_mask)

        inv_w = 1.0 / (w_decay + 1e-6)
        S_t = (
            S_t - vv[:, :, None] * kv[:, None, :] - sav[:, :, None] * bv[:, None, :]
        ) * inv_w[:, None, :]

        dS = dS + dyv[:, :, None] * rv[:, None, :]

        dw = tl.sum(dS * S_t, axis=1) * w_grad_factor
        dk = tl.sum(dS * vv[:, :, None], axis=1)
        dv = tl.sum(dS * kv[:, None, :], axis=2)
        db = tl.sum(dS * sav[:, :, None], axis=1)
        dsa = tl.sum(dS * bv[:, None, :], axis=2)
        da = tl.sum(S_t * dsa[:, :, None], axis=1)

        tl.store(DW + base_ptr_off + t_off, dw.to(DW.dtype.element_ty), mask=ptr_mask)
        tl.store(DK + base_ptr_off + t_off, dk.to(DK.dtype.element_ty), mask=ptr_mask)
        tl.store(DV + base_ptr_off + t_off, dv.to(DV.dtype.element_ty), mask=ptr_mask)
        tl.store(DB + base_ptr_off + t_off, db.to(DB.dtype.element_ty), mask=ptr_mask)
        tl.store(DA + base_ptr_off + t_off, da.to(DA.dtype.element_ty), mask=ptr_mask)

        dS = dS * w_decay[:, None, :] + dsa[:, :, None] * av[:, None, :]

        if t == 0:
            tl.store(
                DH0 + dht_base + row_idx * H_SIZE + col_idx,
                dS.to(DH0.dtype.element_ty),
                mask=state_mask,
            )
