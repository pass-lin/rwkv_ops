import triton
import triton.language as tl


# ======================================================================================
#                                   无 Mask 前向传播
# ======================================================================================
@triton.autotune(
    configs=[
        # 【稳定性修复 1/2】：强制 MINI_BSZ=1
        # 核心原因：限制每个 GPU Block 只处理一个 Batch 样本，避免因 MINI_BSZ > 1
        # 导致在 AMD 显卡上出现超大的寄存器/共享内存分配，从而绕开 ROCm 编译器的
        # 寄存器溢出 (Register Spilling) Bug，这是导致 Segfault 的主要原因之一。
        # 这种做法牺牲了极小的块内并行度，但换来了跨平台的绝对稳定性。
        triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
        # 补偿策略：增加 num_warps 和 num_stages 可以利用更多线程和更深的流水线来
        # 隐藏内存延迟，弥补 MINI_BSZ=1 带来的性能损失。
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def rwkv7_fwd_kernel(
    R,
    W,
    K,
    V,
    A,
    B_param,
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

    # --- 1. 计算 Batch 掩码 ---
    b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
    b_mask = b_range < B_BATCH
    cols = tl.arange(0, H_SIZE)

    # 【稳定性修复 2/2】：强制使用 64 位整数进行指针运算
    # 核心原因：防止在 Batch, T_LEN, N_HEAD, H_SIZE 较大时发生 32 位整数溢出，
    # 导致计算出错误的负数内存地址。同时，这也能规避 AMD 编译器在处理
    # 32 位到 64 位地址扩展时可能存在的 Bug。
    b_range_i64 = b_range.to(tl.int64)
    N_HEAD_i64 = tl.cast(N_HEAD, tl.int64)
    T_LEN_i64 = tl.cast(T_LEN, tl.int64)
    H_SIZE_i64 = tl.cast(H_SIZE, tl.int64)
    pid_h_i64 = tl.cast(pid_h, tl.int64)
    CHUNK_LEN_i64 = tl.cast(CHUNK_LEN, tl.int64)

    # --- 2. 构造各种形状的掩码 ---
    ptr_mask = b_mask[:, None]
    state_mask = b_mask[:, None, None]

    # --- 3. 计算基础偏移 (使用 64 位整数) ---
    base_ptr_off = (
        (b_range_i64[:, None] * N_HEAD_i64 * T_LEN_i64 * H_SIZE_i64)
        + (pid_h_i64 * T_LEN_i64 * H_SIZE_i64)
        + cols[None, :]
    )
    h0_base = (b_range_i64[:, None, None] * N_HEAD_i64 * H_SIZE_i64 * H_SIZE_i64) + (
        pid_h_i64 * H_SIZE_i64 * H_SIZE_i64
    )

    # --- 4. 初始化状态加载 ---
    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]
    state = tl.load(
        H0 + h0_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)

    for t in range(0, T_LEN):
        t_off = t * H_SIZE

        # a. 向量加载
        rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )

        # b. 数学计算
        w_decay = tl.exp(-tl.exp(wv))
        sa_vec = tl.sum(state * av[:, None, :], axis=2)
        tl.store(
            SA_OUT + base_ptr_off + t_off,
            sa_vec.to(SA_OUT.dtype.element_ty),
            mask=ptr_mask,
        )

        # c. 状态演进
        state = (
            state * w_decay[:, None, :]
            + sa_vec[:, :, None] * bv[:, None, :]
            + vv[:, :, None] * kv[:, None, :]
        )

        # d. 计算输出
        y_vec = tl.sum(state * rv[:, None, :], axis=2)
        tl.store(
            OUT + base_ptr_off + t_off, y_vec.to(OUT.dtype.element_ty), mask=ptr_mask
        )

        # e. 状态快照
        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (tl.cast(t, tl.int64) + 1) // CHUNK_LEN_i64 - 1
            chunk_num_i64 = T_LEN_i64 // CHUNK_LEN_i64
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


# ======================================================================================
#                                   无 Mask 反向传播
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
def rwkv7_bwd_kernel(
    R,
    W,
    K,
    V,
    A,
    B_param,
    SA,
    STATE_CHKP,
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
            chunk_num_i64 = T_LEN_i64 // CHUNK_LEN_i64
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
#                                   带 Mask 前向传播
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
def rwkv7_fwd_kernel_with_mask(
    R,
    W,
    K,
    V,
    A,
    B_param,
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
    mask_ptr_base = b_range_i64 * T_LEN_i64

    h0_base = (b_range_i64[:, None, None] * N_HEAD_i64 * H_SIZE_i64 * H_SIZE_i64) + (
        pid_h_i64 * H_SIZE_i64 * H_SIZE_i64
    )
    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]

    state = tl.load(
        H0 + h0_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)

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

        m_val = tl.load(MASK + mask_ptr_base + t, mask=b_mask, other=0.0).to(tl.float32)
        m_3d = m_val[:, None, None]

        w_decay = tl.exp(-tl.exp(wv))
        sa_vec = tl.sum(state * av[:, None, :], axis=2)
        tl.store(
            SA_OUT + base_ptr_off + t_off,
            sa_vec.to(SA_OUT.dtype.element_ty),
            mask=ptr_mask,
        )

        state_cand = (
            state * w_decay[:, None, :]
            + sa_vec[:, :, None] * bv[:, None, :]
            + vv[:, :, None] * kv[:, None, :]
        )

        y_vec = tl.sum(state_cand * rv[:, None, :], axis=2)
        tl.store(
            OUT + base_ptr_off + t_off, y_vec.to(OUT.dtype.element_ty), mask=ptr_mask
        )

        state = m_3d * state_cand + (1.0 - m_3d) * state

        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (tl.cast(t, tl.int64) + 1) // CHUNK_LEN_i64 - 1
            chunk_num_i64 = T_LEN_i64 // CHUNK_LEN_i64
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


# ======================================================================================
#                                   带 Mask 反向传播
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
def rwkv7_bwd_kernel_with_mask(
    R,
    W,
    K,
    V,
    A,
    B_param,
    MASK,
    SA,
    STATE_CHKP,
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
    mask_ptr_base = b_range_i64 * T_LEN_i64

    dht_base = (b_range_i64[:, None, None] * N_HEAD_i64 * H_SIZE_i64 * H_SIZE_i64) + (
        pid_h_i64 * H_SIZE_i64 * H_SIZE_i64
    )
    dS = tl.load(
        DHT + dht_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)

    S_t = tl.zeros([MINI_BSZ, H_SIZE, H_SIZE], dtype=tl.float32)

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

        m_val = tl.load(MASK + mask_ptr_base + t, mask=b_mask, other=0.0).to(tl.float32)
        m_3d = m_val[:, None, None]

        w_decay = tl.exp(-tl.exp(wv))
        w_grad_factor = w_decay * (-tl.exp(wv))

        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (tl.cast(t, tl.int64) + 1) // CHUNK_LEN_i64 - 1
            chunk_num_i64 = T_LEN_i64 // CHUNK_LEN_i64
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

        s_cand_for_dr = (
            S_t * w_decay[:, None, :]
            + sav[:, :, None] * bv[:, None, :]
            + vv[:, :, None] * kv[:, None, :]
        )
        s_for_dr = m_3d * S_t + (1.0 - m_3d) * s_cand_for_dr
        dr = tl.sum(s_for_dr * dyv[:, :, None], axis=1)
        tl.store(DR + base_ptr_off + t_off, dr.to(DR.dtype.element_ty), mask=ptr_mask)

        dS_curr = dyv[:, :, None] * rv[:, None, :]
        dS = dS + dS_curr
        dS_old = dS

        inv_w = 1.0 / (w_decay + 1e-6)
        S_prev = (
            S_t - vv[:, :, None] * kv[:, None, :] - sav[:, :, None] * bv[:, None, :]
        ) * inv_w[:, None, :]
        S_t = m_3d * S_prev + (1.0 - m_3d) * S_t

        dS_param = m_3d * dS_old + (1.0 - m_3d) * dS_curr

        dw = tl.sum(dS_param * S_t, axis=1) * w_grad_factor
        dk = tl.sum(dS_param * vv[:, :, None], axis=1)
        dv = tl.sum(dS_param * kv[:, None, :], axis=2)
        db = tl.sum(dS_param * sav[:, :, None], axis=1)
        dsa = tl.sum(dS_param * bv[:, None, :], axis=2)
        da = tl.sum(S_t * dsa[:, :, None], axis=1)

        tl.store(DW + base_ptr_off + t_off, dw.to(DW.dtype.element_ty), mask=ptr_mask)
        tl.store(DK + base_ptr_off + t_off, dk.to(DK.dtype.element_ty), mask=ptr_mask)
        tl.store(DV + base_ptr_off + t_off, dv.to(DV.dtype.element_ty), mask=ptr_mask)
        tl.store(DB + base_ptr_off + t_off, db.to(DB.dtype.element_ty), mask=ptr_mask)
        tl.store(DA + base_ptr_off + t_off, da.to(DA.dtype.element_ty), mask=ptr_mask)

        trans = dS_param * w_decay[:, None, :] + dsa[:, :, None] * av[:, None, :]
        penetration = dS_old - dS_param
        dS = trans + (1.0 - m_3d) * penetration

        if t == 0:
            tl.store(
                DH0 + dht_base + row_idx * H_SIZE + col_idx,
                dS.to(DH0.dtype.element_ty),
                mask=state_mask,
            )
