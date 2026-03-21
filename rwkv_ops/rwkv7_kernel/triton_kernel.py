import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"MINI_BSZ": bt_num}, num_warps=num_warps, num_stages=num_stages)
        for bt_num in [1, 2, 4, 8]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
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
    # b_mask 形状为 [MINI_BSZ]
    b_mask = b_range < B_BATCH

    cols = tl.arange(0, H_SIZE)

    # --- 2. 构造各种形状的掩码 ---
    # 用于加载向量的掩码 [MINI_BSZ, H_SIZE]
    ptr_mask = b_mask[:, None]
    # 用于加载/存储状态矩阵的掩码 [MINI_BSZ, H_SIZE, H_SIZE]
    state_mask = b_mask[:, None, None]

    # 指针基础偏移
    base_ptr_off = (
        (b_range[:, None] * N_HEAD * T_LEN * H_SIZE)
        + (pid_h * T_LEN * H_SIZE)
        + cols[None, :]
    )

    # 3. 初始化状态加载 (带 Mask)
    h0_base = (b_range[:, None, None] * N_HEAD * H_SIZE * H_SIZE) + (
        pid_h * H_SIZE * H_SIZE
    )
    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]

    # 使用 mask=state_mask，越界处填充 0.0
    state = tl.load(
        H0 + h0_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)

    for t in range(0, T_LEN):
        t_off = t * H_SIZE

        # --- a. 向量加载 (带 Mask) ---
        rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )

        # --- b. 数学计算 ---
        w_decay = tl.exp(-tl.exp(wv))
        sa_vec = tl.sum(state * av[:, None, :], axis=2)

        # 写入 sa_out (带 Mask)
        tl.store(
            SA_OUT + base_ptr_off + t_off,
            sa_vec.to(SA_OUT.dtype.element_ty),
            mask=ptr_mask,
        )

        # --- c. 状态演进 ---
        state = (
            state * w_decay[:, None, :]
            + sa_vec[:, :, None] * bv[:, None, :]
            + vv[:, :, None] * kv[:, None, :]
        )

        # --- d. 计算输出 (带 Mask) ---
        y_vec = tl.sum(state * rv[:, None, :], axis=2)
        tl.store(
            OUT + base_ptr_off + t_off, y_vec.to(OUT.dtype.element_ty), mask=ptr_mask
        )

        # --- e. 状态快照 (带 Mask) ---
        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (t + 1) // CHUNK_LEN - 1
            chkp_base = (
                b_range[:, None, None] * N_HEAD * (T_LEN // CHUNK_LEN) * H_SIZE * H_SIZE
                + pid_h * (T_LEN // CHUNK_LEN) * H_SIZE * H_SIZE
                + chkp_t * (H_SIZE * H_SIZE)
            )
            tl.store(
                STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                state,
                mask=state_mask,
            )


@triton.autotune(
    configs=[
        triton.Config(
            {"MINI_BSZ": bt_num},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for bt_num in [1, 2, 4, 8]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=["H_SIZE"],
)
@triton.jit
def rwkv7_bwd_kernel(
    # --- 前向输入与中间变量 ---
    R,
    W,
    K,
    V,
    A,
    B_param,
    SA,  # 前向保存的 state @ a
    STATE_CHKP,  # 前向每 16 步保存的状态快照
    # --- 维度信息 ---
    B_BATCH,
    N_HEAD,
    T_LEN,
    # --- 梯度输入 ---
    DY,  # 损失对输出 y 的梯度
    DHT,  # 损失对最后一个状态 h_T 的梯度
    # --- 梯度输出 ---
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

    # 1. 确定当前处理的 Batch 范围与掩码
    b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
    b_mask = b_range < B_BATCH
    ptr_mask = b_mask[:, None]

    cols = tl.arange(0, H_SIZE)
    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]
    state_mask = b_mask[:, None, None]

    # 基础内存偏移量 (指向 t=0)
    base_ptr_off = (
        (b_range[:, None] * N_HEAD * T_LEN * H_SIZE)
        + (pid_h * T_LEN * H_SIZE)
        + cols[None, :]
    )

    # 2. 初始化梯度状态 dS [MINI_BSZ, H_SIZE, H_SIZE]
    # 如果外部传入了 dht (即对最终状态的梯度)，则加载它；否则初始化为 0
    dht_base = (b_range[:, None, None] * N_HEAD * H_SIZE * H_SIZE) + (
        pid_h * H_SIZE * H_SIZE
    )
    dS = tl.load(
        DHT + dht_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)

    # S_t 用于在逆推中保存当前状态
    S_t = tl.zeros([MINI_BSZ, H_SIZE, H_SIZE], dtype=tl.float32)

    # 3. 沿时间轴逆向传播 (从 T_LEN-1 到 0)
    for t in range(T_LEN - 1, -1, -1):
        t_off = t * H_SIZE

        # --- a. 加载当前步的向量 ---
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

        # 衰减因子与导数链式乘子
        w_decay = tl.exp(-tl.exp(wv))
        w_grad_factor = w_decay * (-tl.exp(wv))

        # --- b. 状态恢复 (State Recovery) ---
        # 如果是 CHUNK 边界，从显存加载精确的 S_t
        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (t + 1) // CHUNK_LEN - 1
            chkp_base = (
                b_range[:, None, None] * N_HEAD * (T_LEN // CHUNK_LEN) * H_SIZE * H_SIZE
                + pid_h * (T_LEN // CHUNK_LEN) * H_SIZE * H_SIZE
                + chkp_t * (H_SIZE * H_SIZE)
            )
            S_t = tl.load(
                STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                mask=state_mask,
                other=0.0,
            ).to(tl.float32)

        # 计算当前输出对 R (Query) 的梯度： dr = S_t^T * dy
        # S_t 形状 [B, row, col], dyv 形状 [B, row] -> 对 row(axis=1) 求和
        dr = tl.sum(S_t * dyv[:, :, None], axis=1)
        tl.store(DR + base_ptr_off + t_off, dr.to(DR.dtype.element_ty), mask=ptr_mask)

        # 逆推求 S_{t-1}
        # S_{t-1} = (S_t - v*k^T - sa*b^T) / w_decay
        # 注意除以极小值防止除零溢出 (0.000001f 对应 CUDA 代码)
        inv_w = 1.0 / (w_decay + 1e-6)
        S_t = (
            S_t - vv[:, :, None] * kv[:, None, :] - sav[:, :, None] * bv[:, None, :]
        ) * inv_w[:, None, :]

        # --- c. 累加输出梯度到状态梯度 ---
        # dS += dy * r^T
        dS = dS + dyv[:, :, None] * rv[:, None, :]

        # --- d. 计算参数梯度 ---
        # dw = sum_rows(dS * S_{t-1}) * w_grad_factor
        dw = tl.sum(dS * S_t, axis=1) * w_grad_factor

        # dk = dS^T * v (按 axis=1 对齐求和)
        dk = tl.sum(dS * vv[:, :, None], axis=1)

        # dv = dS * k (按 axis=2 对齐求和)
        dv = tl.sum(dS * kv[:, None, :], axis=2)

        # db = dS^T * sa
        db = tl.sum(dS * sav[:, :, None], axis=1)

        # dsa = dS * b
        dsa = tl.sum(dS * bv[:, None, :], axis=2)

        # da = S_{t-1}^T * dsa
        da = tl.sum(S_t * dsa[:, :, None], axis=1)

        # 写入显存
        tl.store(DW + base_ptr_off + t_off, dw.to(DW.dtype.element_ty), mask=ptr_mask)
        tl.store(DK + base_ptr_off + t_off, dk.to(DK.dtype.element_ty), mask=ptr_mask)
        tl.store(DV + base_ptr_off + t_off, dv.to(DV.dtype.element_ty), mask=ptr_mask)
        tl.store(DB + base_ptr_off + t_off, db.to(DB.dtype.element_ty), mask=ptr_mask)
        tl.store(DA + base_ptr_off + t_off, da.to(DA.dtype.element_ty), mask=ptr_mask)

        # --- e. 状态梯度传导至 t-1 ---
        # dS_{t-1} = dS_t * w_decay^T + dsa * a^T
        dS = dS * w_decay[:, None, :] + dsa[:, :, None] * av[:, None, :]

        # 如果逆推到了起点，保存对 h0 的梯度
        if t == 0:
            tl.store(
                DH0 + dht_base + row_idx * H_SIZE + col_idx,
                dS.to(DH0.dtype.element_ty),
                mask=state_mask,
            )


# ====================================================================
# 带 Mask 的前向传播内核
# ====================================================================
@triton.autotune(
    configs=[
        triton.Config({"MINI_BSZ": bt_num}, num_warps=num_warps, num_stages=num_stages)
        for bt_num in [1, 2, 4, 8]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
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
    H0,  # 新增 MASK 参数
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

    # 1. 计算 Batch 掩码与偏移
    b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
    b_mask = b_range < B_BATCH
    cols = tl.arange(0, H_SIZE)

    ptr_mask = b_mask[:, None]
    state_mask = b_mask[:, None, None]

    base_ptr_off = (
        (b_range[:, None] * N_HEAD * T_LEN * H_SIZE)
        + (pid_h * T_LEN * H_SIZE)
        + cols[None, :]
    )

    # Mask 张量形状为 [B, T]，计算基础偏移
    mask_ptr_base = b_range * T_LEN

    # 2. 初始化状态
    h0_base = (b_range[:, None, None] * N_HEAD * H_SIZE * H_SIZE) + (
        pid_h * H_SIZE * H_SIZE
    )
    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]

    state = tl.load(
        H0 + h0_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)

    for t in range(0, T_LEN):
        t_off = t * H_SIZE

        # a. 向量与 Mask 加载
        rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )

        # 加载时间步 t 的掩码 [MINI_BSZ] -> 扩展为 [MINI_BSZ, 1, 1]
        m_val = tl.load(MASK + mask_ptr_base + t, mask=b_mask, other=0.0).to(tl.float32)
        m_3d = m_val[:, None, None]

        # b. 计算候选状态 (Candidate State)
        w_decay = tl.exp(-tl.exp(wv))
        sa_vec = tl.sum(state * av[:, None, :], axis=2)

        tl.store(
            SA_OUT + base_ptr_off + t_off,
            sa_vec.to(SA_OUT.dtype.element_ty),
            mask=ptr_mask,
        )

        # state_cand 对应 CUDA 代码中的 s_cand
        state_cand = (
            state * w_decay[:, None, :]
            + sa_vec[:, :, None] * bv[:, None, :]
            + vv[:, :, None] * kv[:, None, :]
        )

        # c. 计算输出 (输出基于 state_cand)
        y_vec = tl.sum(state_cand * rv[:, None, :], axis=2)
        tl.store(
            OUT + base_ptr_off + t_off, y_vec.to(OUT.dtype.element_ty), mask=ptr_mask
        )

        # d. 状态过滤 (State Gating)
        # m=1 采用新状态, m=0 保持旧状态
        state = m_3d * state_cand + (1.0 - m_3d) * state

        # e. 状态快照
        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (t + 1) // CHUNK_LEN - 1
            chkp_base = (
                b_range[:, None, None] * N_HEAD * (T_LEN // CHUNK_LEN) * H_SIZE * H_SIZE
                + pid_h * (T_LEN // CHUNK_LEN) * H_SIZE * H_SIZE
                + chkp_t * (H_SIZE * H_SIZE)
            )
            tl.store(
                STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                state,
                mask=state_mask,
            )


# ====================================================================
# 带 Mask 的反向传播内核
# ====================================================================
@triton.autotune(
    configs=[
        triton.Config({"MINI_BSZ": bt_num}, num_warps=num_warps, num_stages=num_stages)
        for bt_num in [1, 2, 4, 8]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
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

    ptr_mask = b_mask[:, None]
    state_mask = b_mask[:, None, None]

    base_ptr_off = (
        (b_range[:, None] * N_HEAD * T_LEN * H_SIZE)
        + (pid_h * T_LEN * H_SIZE)
        + cols[None, :]
    )
    mask_ptr_base = b_range * T_LEN

    dht_base = (b_range[:, None, None] * N_HEAD * H_SIZE * H_SIZE) + (
        pid_h * H_SIZE * H_SIZE
    )
    dS = tl.load(
        DHT + dht_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)

    S_t = tl.zeros([MINI_BSZ, H_SIZE, H_SIZE], dtype=tl.float32)

    for t in range(T_LEN - 1, -1, -1):
        t_off = t * H_SIZE

        # a. 加载向量与 Mask
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

        # b. 状态恢复 (State Recovery)
        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (t + 1) // CHUNK_LEN - 1
            chkp_base = (
                b_range[:, None, None] * N_HEAD * (T_LEN // CHUNK_LEN) * H_SIZE * H_SIZE
                + pid_h * (T_LEN // CHUNK_LEN) * H_SIZE * H_SIZE
                + chkp_t * (H_SIZE * H_SIZE)
            )
            S_t = tl.load(
                STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                mask=state_mask,
                other=0.0,
            ).to(tl.float32)

        # 【修复1】对 R (Query) 的梯度 dr
        # 如果 m=1，前向中 y_t 使用的是新计算的 state_cand，但由于 S_t 已经被更新，此时 S_t 就是 state_cand。
        # 如果 m=0，前向中 y_t 依然使用 state_cand 进行计算！而此时的 S_t 保持为 S_{t-1}。
        # 因此，我们需要逆推一下 s_cand 来计算 m=0 时的梯度。
        s_cand_for_dr = (
            S_t * w_decay[:, None, :]
            + sav[:, :, None] * bv[:, None, :]
            + vv[:, :, None] * kv[:, None, :]
        )
        # 根据 Mask 决定：m=1 用真实的 S_t，m=0 用基于当前 S_t (即 S_{t-1}) 推出来的 s_cand
        s_for_dr = m_3d * S_t + (1.0 - m_3d) * s_cand_for_dr

        dr = tl.sum(s_for_dr * dyv[:, :, None], axis=1)
        tl.store(DR + base_ptr_off + t_off, dr.to(DR.dtype.element_ty), mask=ptr_mask)

        # 【修复2/3】累加当前输出梯度到总梯度，并保存完整梯度 (包含未来穿透)
        dS_curr = dyv[:, :, None] * rv[:, None, :]
        dS = dS + dS_curr
        dS_old = dS  # 保存带有未来梯度的老 dS

        # 【修复4】逆推 S_{t-1}
        # 仅当 m=1 时需要逆推，m=0 时 S_t 本就是 S_{t-1}
        inv_w = 1.0 / (w_decay + 1e-6)
        S_prev = (
            S_t - vv[:, :, None] * kv[:, None, :] - sav[:, :, None] * bv[:, None, :]
        ) * inv_w[:, None, :]

        S_t = m_3d * S_prev + (1.0 - m_3d) * S_t

        # 【修复5】分离参数计算用的梯度 dS_param
        # m=1: 使用完整梯度 (包含未来)。m=0: 仅用当前步的输出梯度 (未来穿透不影响参数)
        dS_param = m_3d * dS_old + (1.0 - m_3d) * dS_curr

        # 【修复6】参数梯度计算
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

        # 【修复7】状态梯度传导至 t-1
        # 标准 RNN 回传路径 (trans)
        trans = dS_param * w_decay[:, None, :] + dsa[:, :, None] * av[:, None, :]

        # 穿透路径 (仅当 m=0 时存在)
        penetration = dS_old - dS_param

        # 关键合并：always 保留 trans，m=0 时额外加上穿透部分
        dS = trans + (1.0 - m_3d) * penetration

        # 如果逆推到了起点，保存对 h0 的梯度
        if t == 0:
            tl.store(
                DH0 + dht_base + row_idx * H_SIZE + col_idx,
                dS.to(DH0.dtype.element_ty),
                mask=state_mask,
            )
