"""mHC Pre-Op 共享 Triton 内核（Sinkhorn-Knopp + Stream Aggregate）。"""

import triton
import triton.language as tl


# mHC Pre-Op 前向 Triton kernel。
#
# 每个 program 处理 (BLOCK_BT, BLOCK_C) 个 (batch*time, channel) 元素：
# pid_c == 0 的 program 负责 Sinkhorn-Knopp 生成双随机矩阵；
# 所有 program 负责 Stream Aggregate 的加权求和。
#
# Args:
#   x_ptr: [Total_BT, n, C], bfloat16, row-major。多流输入。
#   h_res_in_ptr: [Total_BT, n, n], float32, row-major。未归一化残差矩阵。
#   h_pre_in_ptr: [Total_BT, n], float32, row-major。未激活聚合权重。
#   out_ptr: [Total_BT, C], bfloat16, row-major。聚合输出。
#   H_res_out_ptr: [Total_BT, n, n], float32, row-major。双随机残差矩阵。
#
# 编译期宏:
#   Total_BT_CONST: batch * time。
#   NSIZE: 流数量 n。
#   CSIZE: 通道数 C。
#   NUM_ITERS: Sinkhorn-Knopp 迭代次数。
#   EPS: 数值稳定常数。
#   BLOCK_BT, BLOCK_C: tile 大小。
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_BT": bt_num, "BLOCK_C": csize},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for bt_num in [16, 32]
        for csize in [128, 256, 512]
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["CSIZE", "Total_BT_CONST"],
)
@triton.jit
def sinkhorn_aggregate_fused_kernel(
    x_ptr,
    h_res_in_ptr,
    h_pre_in_ptr,
    out_ptr,
    H_res_out_ptr,
    Total_BT_CONST: tl.constexpr,
    NSIZE: tl.constexpr,
    CSIZE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
    EPS: tl.constexpr,
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
    BLOCK_BT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_bt = pid_bt * BLOCK_BT + tl.arange(0, BLOCK_BT)
    mask_bt = offs_bt < Total_BT_CONST

    # Sinkhorn-Knopp 只在 pid_c == 0 时执行，避免重复写回。
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

    # Stream Aggregate
    # 加载聚合权重 (一次性加载所有流的权重，效率最高)
    offs_n = tl.arange(0, NSIZE)
    h_pre_ptr_loc = (
        h_pre_in_ptr
        + offs_bt[:, None] * stride_h_pre_in_bt
        + offs_n[None, :] * stride_h_pre_in_n
    )

    h_pre = tl.load(h_pre_ptr_loc, mask=mask_bt[:, None], other=0.0).to(tl.float32)
    w_pre = tl.sigmoid(h_pre)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < CSIZE

    acc = tl.zeros([BLOCK_BT, BLOCK_C], dtype=tl.float32)

    n_range = tl.arange(0, NSIZE)[None, :]

    for k in tl.static_range(NSIZE):
        # 用掩码从 [BLOCK_BT, N] 权重中提取第 k 列，避免 gather。
        col_mask = n_range == k
        w_k = tl.sum(w_pre * col_mask, axis=1)[:, None]

        x_ptr_k = (
            x_ptr
            + offs_bt[:, None] * stride_x_bt
            + k * stride_x_n
            + offs_c[None, :] * stride_x_c
        )

        load_mask = mask_bt[:, None] & mask_c[None, :]
        val_x = tl.load(x_ptr_k, mask=load_mask, other=0.0).to(tl.float32)

        acc += val_x * w_k

    out_loc = (
        out_ptr + offs_bt[:, None] * stride_out_bt + offs_c[None, :] * stride_out_c
    )

    store_mask = mask_bt[:, None] & mask_c[None, :]
    tl.store(out_loc, acc.to(tl.bfloat16), mask=store_mask)


# mHC Pre-Op 反向 Triton kernel。
#
# 每个 program 处理一个 batch*time 实例，沿 Channel 维度循环分块。
# Sinkhorn 梯度通过重算前向 P 并执行 VJP 逆向迭代得到；
# Aggregate 梯度在 persistent 寄存器中累加，避免 atomic_add。
#
# Args:
#   grad_out_ptr: [Total_BT, C], bfloat16, row-major。上游传给 out 的梯度。
#   grad_H_res_out_ptr: [Total_BT, n, n], float32, row-major。上游传给 H_res 的梯度。
#   x_ptr: [Total_BT, n, C], bfloat16, row-major。前向多流输入。
#   h_res_in_ptr: [Total_BT, n, n], float32, row-major。前向未归一化残差矩阵。
#   h_pre_in_ptr: [Total_BT, n], float32, row-major。前向未激活聚合权重。
#   grad_x_ptr: [Total_BT, n, C], bfloat16, row-major。x 的梯度输出。
#   grad_h_res_in_ptr: [Total_BT, n, n], float32, row-major。h_res 的梯度输出。
#   grad_h_pre_in_ptr: [Total_BT, n], float32, row-major。h_pre 的梯度输出。
#
# 编译期宏:
#   TOTAL_BT_CONST, NSIZE, CHANNEL_SIZE, NUM_ITERS, EPS。
#   BLOCK_CHANNEL: channel tile 大小。
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
    grad_out_ptr,  # [Total_BT, C]
    grad_H_res_out_ptr,  # [Total_BT, n, n]
    x_ptr,  # [Total_BT, n, C]
    h_res_in_ptr,  # [Total_BT, n, n]
    h_pre_in_ptr,  # [Total_BT, n]
    grad_x_ptr,  # [Total_BT, n, C]
    grad_h_res_in_ptr,  # [Total_BT, n, n]
    grad_h_pre_in_ptr,  # [Total_BT, n]
    TOTAL_BT_CONST: tl.constexpr,
    NSIZE: tl.constexpr,
    CHANNEL_SIZE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
    EPS: tl.constexpr,
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
    BLOCK_CHANNEL: tl.constexpr,
):
    # Grid Y = 1 保证单个 program 处理整行 C，从而消除原子加。
    pid_bt = tl.program_id(0)

    off_n1 = tl.arange(0, NSIZE)
    off_n2 = tl.arange(0, NSIZE)

    h_res_ptr = (
        h_res_in_ptr
        + pid_bt * stride_h_res_bt
        + off_n1[:, None] * stride_h_res_n1
        + off_n2[None, :] * stride_h_res_n2
    )
    h_res = tl.load(h_res_ptr).to(tl.float32)

    # 手写 VJP 强制重算前向 P，前向只保存原始输入以省显存。
    max_val = tl.max(tl.max(h_res, 1), 0)
    P = tl.exp(h_res - max_val)
    for _ in tl.range(NUM_ITERS):
        P /= tl.sum(P, axis=1)[:, None] + EPS
        P /= tl.sum(P, axis=0)[None, :] + EPS

    gH_ptr = (
        grad_H_res_out_ptr
        + pid_bt * stride_gH_bt
        + off_n1[:, None] * stride_gH_n1
        + off_n2[None, :] * stride_gH_n2
    )
    dP = tl.load(gH_ptr).to(tl.float32)

    for _ in tl.static_range(NUM_ITERS):
        # 逆向列归一化：dX = dY - Y * sum(dY * Y)
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

    off_n = tl.arange(0, NSIZE)
    h_pre_ptr = h_pre_in_ptr + pid_bt * stride_h_pre_bt + off_n * stride_h_pre_n
    h_pre = tl.load(h_pre_ptr).to(tl.float32)
    w_pre = tl.sigmoid(h_pre)
    dw_pre = w_pre * (1.0 - w_pre)
    # 持久化累加器
    acc_gh_pre = tl.zeros([NSIZE], dtype=tl.float32)

    for start_c in tl.static_range(0, CHANNEL_SIZE, BLOCK_CHANNEL):
        off_c = start_c + tl.arange(0, BLOCK_CHANNEL)
        mask_c = off_c < CHANNEL_SIZE

        gout_ptr = grad_out_ptr + pid_bt * stride_gout_bt + off_c * stride_gout_c
        g_out = tl.load(gout_ptr, mask=mask_c, other=0.0).to(tl.float32)

        for k in tl.static_range(NSIZE):
            w_k = tl.sum(w_pre * (off_n == k), axis=0)

            gx_chunk = g_out * w_k
            gx_out_ptr = (
                grad_x_ptr
                + pid_bt * stride_gx_bt
                + k * stride_gx_n
                + off_c * stride_gx_c
            )
            tl.store(gx_out_ptr, gx_chunk.to(tl.bfloat16), mask=mask_c)

            x_ptr_k = x_ptr + pid_bt * stride_x_bt + k * stride_x_n + off_c * stride_x_c
            x_chunk = tl.load(x_ptr_k, mask=mask_c, other=0.0).to(tl.float32)

            dot_sum = tl.sum(g_out * x_chunk, axis=0)
            acc_gh_pre += dot_sum * (off_n == k)

    final_gh_pre = acc_gh_pre * dw_pre
    gh_pre_out_ptr = (
        grad_h_pre_in_ptr + pid_bt * stride_gh_pre_bt + off_n * stride_gh_pre_n
    )
    tl.store(gh_pre_out_ptr, final_gh_pre)
