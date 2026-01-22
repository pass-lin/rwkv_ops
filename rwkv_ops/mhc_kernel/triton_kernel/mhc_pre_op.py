import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_BT": bt_num, "BLOCK_C": csize},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for bt_num in [16,32]
        for csize in [128, 256, 512]
        for num_warps in [4, 8]
        for num_stages in [2, 3,4]
    ],
    key=["CSIZE", "Total_BT_CONST"],
)
@triton.jit
def sinkhorn_aggregate_fused_kernel(
    # --- 1. 指针参数 ---
    x_ptr,
    h_res_in_ptr,
    h_pre_in_ptr,
    out_ptr,
    H_res_out_ptr,
    # --- 2. 编译时常量 ---
    Total_BT_CONST: tl.constexpr,
    NSIZE: tl.constexpr,
    CSIZE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
    EPS: tl.constexpr,
    # --- 3. 步幅参数 (constexpr) ---
    stride_x_bt: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_c: tl.constexpr,
    stride_h_res_in_bt: tl.constexpr,
    stride_h_res_in_n1: tl.constexpr,
    stride_h_res_in_n2: tl.constexpr,
    stride_h_pre_in_bt: tl.constexpr,
    stride_h_pre_in_n: tl.constexpr,
    stride_out_bt: tl.constexpr,
    stride_out_c: tl.constexpr,
    stride_Hr_out_bt: tl.constexpr,
    stride_Hr_out_n1: tl.constexpr,
    stride_Hr_out_n2: tl.constexpr,
    # --- 4. 调优参数 ---
    BLOCK_BT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_bt = pid_bt * BLOCK_BT + tl.arange(0, BLOCK_BT)
    mask_bt = offs_bt < Total_BT_CONST

    # -----------------------------------------------------------
    # PART 1: Sinkhorn-Knopp (仅 pid_c == 0 执行)
    # -----------------------------------------------------------
    if pid_c == 0:
        offs_n1 = tl.arange(0, NSIZE)
        offs_n2 = tl.arange(0, NSIZE)

        h_res_ptr_loc = (
            h_res_in_ptr
            + offs_bt[:, None, None] * stride_h_res_in_bt
            + offs_n1[None, :, None] * stride_h_res_in_n1
            + offs_n2[None, None, :] * stride_h_res_in_n2
        )

        h_res = tl.load(h_res_ptr_loc, mask=mask_bt[:, None, None], other=0.0).to(
            tl.float32
        )

        max_val = tl.max(h_res, axis=2)
        max_val = tl.max(max_val, axis=1)
        h_res_stabilized = h_res - max_val[:, None, None]
        P = tl.exp(h_res_stabilized)

        for _ in tl.static_range(NUM_ITERS):
            row_sum = tl.sum(P, axis=2)
            P = P / (row_sum[:, :, None] + EPS)
            col_sum = tl.sum(P, axis=1)
            P = P / (col_sum[:, None, :] + EPS)

        H_out_loc = (
            H_res_out_ptr
            + offs_bt[:, None, None] * stride_Hr_out_bt
            + offs_n1[None, :, None] * stride_Hr_out_n1
            + offs_n2[None, None, :] * stride_Hr_out_n2
        )
        tl.store(H_out_loc, P, mask=mask_bt[:, None, None])

    # -----------------------------------------------------------
    # PART 2: Stream Aggregate
    # -----------------------------------------------------------

    # 2.1 加载聚合权重 (一次性加载所有流的权重，效率最高)
    offs_n = tl.arange(0, NSIZE)
    h_pre_ptr_loc = (
        h_pre_in_ptr
        + offs_bt[:, None] * stride_h_pre_in_bt
        + offs_n[None, :] * stride_h_pre_in_n
    )

    h_pre = tl.load(h_pre_ptr_loc, mask=mask_bt[:, None], other=0.0).to(tl.float32)
    w_pre = tl.sigmoid(h_pre)  # [BLOCK_BT, N]

    # 2.2 聚合计算
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < CSIZE

    acc = tl.zeros([BLOCK_BT, BLOCK_C], dtype=tl.float32)

    # 用于列提取的辅助索引 [1, N]
    n_range = tl.arange(0, NSIZE)[None, :]

    for k in tl.static_range(NSIZE):
        # --- 关键修改: 使用 Masking 提取第 k 列 ---
        # w_pre: [BLOCK_BT, N]
        # col_mask: [1, N] (例如 k=2 -> [0,0,1,0])
        col_mask = n_range == k

        # 1. 广播乘法: 保留第 k 列，其他置零
        # 2. Sum(axis=1): 降维成 [BLOCK_BT]
        # 3. [:, None]: 广播成 [BLOCK_BT, 1] 用于后续乘法
        w_k = tl.sum(w_pre * col_mask, axis=1)[:, None]

        # 加载 x
        x_ptr_k = (
            x_ptr
            + offs_bt[:, None] * stride_x_bt
            + k * stride_x_n
            + offs_c[None, :] * stride_x_c
        )

        load_mask = mask_bt[:, None] & mask_c[None, :]
        val_x = tl.load(x_ptr_k, mask=load_mask, other=0.0).to(tl.float32)

        acc += val_x * w_k

    # 2.3 写回
    out_loc = (
        out_ptr + offs_bt[:, None] * stride_out_bt + offs_c[None, :] * stride_out_c
    )

    store_mask = mask_bt[:, None] & mask_c[None, :]
    tl.store(out_loc, acc.to(tl.bfloat16), mask=store_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_CHANNEL": bc},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for bc in [128, 256, 512]
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["CHANNEL_SIZE", "TOTAL_BT_CONST"],
)
@triton.jit
def sinkhorn_aggregate_bwd_kernel(
    # --- 1. 输入指针 ---
    grad_out_ptr,  # [Total_BT, C]
    grad_H_res_out_ptr,  # [Total_BT, n, n]
    x_ptr,  # [Total_BT, n, C]
    h_res_in_ptr,  # [Total_BT, n, n]
    h_pre_in_ptr,  # [Total_BT, n]
    # --- 2. 输出指针 ---
    grad_x_ptr,  # [Total_BT, n, C]
    grad_h_res_in_ptr,  # [Total_BT, n, n]
    grad_h_pre_in_ptr,  # [Total_BT, n]
    # --- 3. 编译时常量 ---
    TOTAL_BT_CONST: tl.constexpr,
    NSIZE: tl.constexpr,
    CHANNEL_SIZE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
    EPS: tl.constexpr,
    # --- 4. 步幅参数 (constexpr) ---
    stride_gout_bt: tl.constexpr,
    stride_gout_c: tl.constexpr,
    stride_gH_bt: tl.constexpr,
    stride_gH_n1: tl.constexpr,
    stride_gH_n2: tl.constexpr,
    stride_x_bt: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_c: tl.constexpr,
    stride_h_res_bt: tl.constexpr,
    stride_h_res_n1: tl.constexpr,
    stride_h_res_n2: tl.constexpr,
    stride_h_pre_bt: tl.constexpr,
    stride_h_pre_n: tl.constexpr,
    stride_gx_bt: tl.constexpr,
    stride_gx_n: tl.constexpr,
    stride_gx_c: tl.constexpr,
    stride_gh_res_bt: tl.constexpr,
    stride_gh_res_n1: tl.constexpr,
    stride_gh_res_n2: tl.constexpr,
    stride_gh_pre_bt: tl.constexpr,
    stride_gh_pre_n: tl.constexpr,
    # --- 5. 调优参数 ---
    BLOCK_CHANNEL: tl.constexpr,
):
    # 强制单 Block 处理整行 C 以消除 atomic_add
    pid_bt = tl.program_id(0)

    # ===========================================================
    # PART 1: 精确 Sinkhorn 梯度 (先处理并写回，释放寄存器)
    # ===========================================================
    off_n1 = tl.arange(0, NSIZE)
    off_n2 = tl.arange(0, NSIZE)

    # [1.1] 加载原始输入 h_res_in 并重算前向 P
    h_res_ptr = (
        h_res_in_ptr
        + pid_bt * stride_h_res_bt
        + off_n1[:, None] * stride_h_res_n1
        + off_n2[None, :] * stride_h_res_n2
    )
    h_res = tl.load(h_res_ptr).to(tl.float32)

    # 前向重算 (指数空间)
    max_val = tl.max(tl.max(h_res, 1), 0)
    P = tl.exp(h_res - max_val)
    for _ in tl.range(NUM_ITERS):
        P /= tl.sum(P, axis=1)[:, None] + EPS
        P /= tl.sum(P, axis=0)[None, :] + EPS

    # [1.2] 加载输出梯度并执行精确 VJP 逆向迭代
    gH_ptr = (
        grad_H_res_out_ptr
        + pid_bt * stride_gH_bt
        + off_n1[:, None] * stride_gH_n1
        + off_n2[None, :] * stride_gH_n2
    )
    dP = tl.load(gH_ptr).to(tl.float32)

    for _ in tl.static_range(NUM_ITERS):
        # 逆向列归一化: dX = dY - Y * sum(dY * Y)
        dP = dP - P * tl.sum(dP * P, axis=0)[None, :]
        # 逆向行归一化
        dP = dP - P * tl.sum(dP * P, axis=1)[:, None]

    # 写回 grad_h_res = dP * P (映射回 Log 空间)
    grad_h_res = dP * P
    gh_res_out_ptr = (
        grad_h_res_in_ptr
        + pid_bt * stride_gh_res_bt
        + off_n1[:, None] * stride_gh_res_n1
        + off_n2[None, :] * stride_gh_res_n2
    )
    tl.store(gh_res_out_ptr, grad_h_res)

    # ===========================================================
    # PART 2: Aggregate 梯度 (Persistent Reduction)
    # ===========================================================

    # [2.1] 准备聚合权重与掩码
    off_n = tl.arange(0, NSIZE)
    n_range = off_n[None, :]
    h_pre_ptr = h_pre_in_ptr + pid_bt * stride_h_pre_bt + off_n * stride_h_pre_n
    h_pre = tl.load(h_pre_ptr).to(tl.float32)
    w_pre = tl.sigmoid(h_pre)
    dw_pre = w_pre * (1.0 - w_pre)  # Sigmoid 导数

    # 持久化累加器
    acc_gh_pre = tl.zeros([NSIZE], dtype=tl.float32)

    # [2.2] 跨步遍历 Channel 维度
    for start_c in tl.static_range(0, CHANNEL_SIZE, BLOCK_CHANNEL):
        off_c = start_c + tl.arange(0, BLOCK_CHANNEL)
        mask_c = off_c < CHANNEL_SIZE

        # 加载 grad_out 块
        gout_ptr = grad_out_ptr + pid_bt * stride_gout_bt + off_c * stride_gout_c
        g_out = tl.load(gout_ptr, mask=mask_c, other=0.0).to(tl.float32)

        # 遍历流计算 grad_x 并规约 grad_h_pre
        for k in tl.static_range(NSIZE):
            # 获取当前流权重 (Masking trick)
            w_k = tl.sum(w_pre * (off_n == k), axis=0)

            # A. 计算并存储 grad_x [k, c]
            gx_chunk = g_out * w_k
            gx_out_ptr = (
                grad_x_ptr
                + pid_bt * stride_gx_bt
                + k * stride_gx_n
                + off_c * stride_gx_c
            )
            tl.store(gx_out_ptr, gx_chunk.to(tl.bfloat16), mask=mask_c)

            # B. 计算 grad_h_pre 规约分量
            x_ptr_k = x_ptr + pid_bt * stride_x_bt + k * stride_x_n + off_c * stride_x_c
            x_chunk = tl.load(x_ptr_k, mask=mask_c, other=0.0).to(tl.float32)

            # dot_sum = sum_c(grad_out * x)
            dot_sum = tl.sum(g_out * x_chunk, axis=0)
            acc_gh_pre += dot_sum * (off_n == k)

    # [2.3] 规约结束，应用 Sigmoid 导数并写回
    final_gh_pre = acc_gh_pre * dw_pre
    gh_pre_out_ptr = (
        grad_h_pre_in_ptr + pid_bt * stride_gh_pre_bt + off_n * stride_gh_pre_n
    )
    tl.store(gh_pre_out_ptr, final_gh_pre)
