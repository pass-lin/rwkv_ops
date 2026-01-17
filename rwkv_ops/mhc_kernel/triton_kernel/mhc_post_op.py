import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_CHANNEL": block_c}, num_warps=num_warps, num_stages=num_stages
        )
        for block_c in [64, 128, 256, 512, 1024, 2048]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["CHANNEL_SIZE"],
)
@triton.jit
def mhc_fused_forward_kernel(
    # --- 指针参数 ---
    x_expanded_ptr,
    h_post_raw_ptr,
    H_res_ptr,
    layer_out_ptr,
    output_ptr,
    # --- 步幅参数 (tl.constexpr) ---
    stride_output_batch_time: tl.constexpr,
    stride_output_n_size: tl.constexpr,
    stride_output_channel: tl.constexpr,
    stride_x_batch_time: tl.constexpr,
    stride_x_n_size: tl.constexpr,
    stride_x_channel: tl.constexpr,
    stride_h_batch_time: tl.constexpr,
    stride_h_n_size: tl.constexpr,
    stride_H_batch_time: tl.constexpr,
    stride_H_n_size_1: tl.constexpr,
    stride_H_n_size_2: tl.constexpr,
    stride_layer_out_batch_time: tl.constexpr,
    stride_layer_out_channel: tl.constexpr,
    # --- 编译时常量 ---
    CHANNEL_SIZE: tl.constexpr,
    NSIZE: tl.constexpr,
    BLOCK_CHANNEL: tl.constexpr,
):
    pid_batch_time = tl.program_id(0)
    pid_channel_block = tl.program_id(1)

    offset_channel = pid_channel_block * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    mask_channel = offset_channel < CHANNEL_SIZE

    # 基础指针
    x_expanded_base = x_expanded_ptr + pid_batch_time * stride_x_batch_time
    h_post_raw_base = h_post_raw_ptr + pid_batch_time * stride_h_batch_time
    H_res_base = H_res_ptr + pid_batch_time * stride_H_batch_time
    layer_out_base = layer_out_ptr + pid_batch_time * stride_layer_out_batch_time
    output_base = output_ptr + pid_batch_time * stride_output_batch_time

    # 索引 n: [0, 1, ..., n-1]
    index_n = tl.arange(0, NSIZE)

    # 一次性读取 h_post [NSIZE]
    h_post_values = tl.load(h_post_raw_base + index_n * stride_h_n_size).to(tl.float32)
    weight_values = tl.sigmoid(h_post_values) * 2.0

    # 一次性读取整个 H_res 矩阵 [NSIZE, NSIZE]
    # 使用 2D 索引一次性把小矩阵拉入寄存器a
    offs_h_row = tl.arange(0, NSIZE)
    offs_h_col = tl.arange(0, NSIZE)
    # H_all shape: [NSIZE, NSIZE]

    layer_out_values = tl.load(
        layer_out_base + offset_channel * stride_layer_out_channel,
        mask=mask_channel,
        other=0.0,
    ).to(tl.float32)

    # accumulator = weight * layer_out
    accumulator = weight_values[:, None] * layer_out_values[None, :]

    for k in tl.static_range(NSIZE):
        # A. 加载 x_expanded 的第 k 行 [1, BLOCK_CHANNEL]
        x_row_k_ptr = (
            x_expanded_base + k * stride_x_n_size + offset_channel * stride_x_channel
        )
        x_row_k_value = tl.load(x_row_k_ptr, mask=mask_channel, other=0.0).to(
            tl.float32
        )

        H_column_k_ptr = (
            H_res_base + index_n * stride_H_n_size_1 + k * stride_H_n_size_2
        )
        H_column_k_value = tl.load(H_column_k_ptr).to(tl.float32)

        # C. 累加外积
        # [N, 1] * [1, BLOCK_C] -> [N, BLOCK_C]
        accumulator += H_column_k_value[:, None] * x_row_k_value[None, :]

    index_n_output = index_n[:, None]
    index_c_output = offset_channel[None, :]

    output_target_ptrs = (
        output_base
        + index_n_output * stride_output_n_size
        + index_c_output * stride_output_channel
    )

    tl.store(
        output_target_ptrs, accumulator.to(tl.bfloat16), mask=mask_channel[None, :]
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_CHANNEL": block_c}, num_warps=num_warps, num_stages=num_stages
        )
        for block_c in [64, 128, 256, 512, 1024]
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["CHANNEL_SIZE"],
)
@triton.jit
def mhc_fused_backward_kernel(
    # --- Pointers ---
    x_expanded_ptr,
    h_post_raw_ptr,
    H_res_ptr,
    layer_out_ptr,
    grad_output_ptr,
    grad_x_ptr,
    grad_h_ptr,
    grad_H_ptr,
    grad_layer_out_ptr,
    # --- Strides (tl.constexpr) ---
    stride_x_bt: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_c: tl.constexpr,
    stride_h_bt: tl.constexpr,
    stride_h_n: tl.constexpr,
    stride_H_bt: tl.constexpr,
    stride_H_n1: tl.constexpr,
    stride_H_n2: tl.constexpr,
    stride_l_bt: tl.constexpr,
    stride_l_c: tl.constexpr,
    stride_g_out_bt: tl.constexpr,
    stride_g_out_n: tl.constexpr,
    stride_g_out_c: tl.constexpr,
    stride_grad_x_bt: tl.constexpr,
    stride_grad_x_n: tl.constexpr,
    stride_grad_x_c: tl.constexpr,
    stride_grad_l_bt: tl.constexpr,
    stride_grad_l_c: tl.constexpr,
    stride_grad_h_bt: tl.constexpr,
    stride_grad_h_n: tl.constexpr,
    stride_grad_H_bt: tl.constexpr,
    stride_grad_H_n1: tl.constexpr,
    stride_grad_H_n2: tl.constexpr,
    # --- Constants ---
    CHANNEL_SIZE: tl.constexpr,
    NSIZE: tl.constexpr,
    BLOCK_CHANNEL: tl.constexpr,
):
    # 网格索引
    pid_batch_time = tl.program_id(0)
    pid_channel_block = tl.program_id(1)

    offset_channel = pid_channel_block * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    mask_channel = offset_channel < CHANNEL_SIZE

    # 1. 基础指针计算
    x_base = x_expanded_ptr + pid_batch_time * stride_x_bt
    h_post_base = h_post_raw_ptr + pid_batch_time * stride_h_bt
    H_res_base = H_res_ptr + pid_batch_time * stride_H_bt
    l_base = layer_out_ptr + pid_batch_time * stride_l_bt
    g_out_base = grad_output_ptr + pid_batch_time * stride_g_out_bt

    g_x_base = grad_x_ptr + pid_batch_time * stride_grad_x_bt
    g_l_base = grad_layer_out_ptr + pid_batch_time * stride_grad_l_bt
    g_h_post_base = grad_h_ptr + pid_batch_time * stride_grad_h_bt
    g_H_res_base = grad_H_ptr + pid_batch_time * stride_grad_H_bt

    # -----------------------------------------------------------
    # Step 2: 预加载 Vector h
    # -----------------------------------------------------------
    index_n = tl.arange(0, NSIZE)

    h_vals = tl.load(h_post_base + index_n * stride_h_n).to(tl.float32)
    sig_h = tl.sigmoid(h_vals)
    weight_vals = sig_h * 2.0
    dsig_h = sig_h * (1.0 - sig_h) * 2.0

    # -----------------------------------------------------------
    # Step 3: 加载流式数据块 (Grad Output & Layer Out)
    # -----------------------------------------------------------
    # 加载全量 grad_output (用于 grad_l, grad_h, grad_H)
    g_out_vals = tl.load(
        g_out_base
        + index_n[:, None] * stride_g_out_n
        + offset_channel[None, :] * stride_g_out_c,
        mask=mask_channel[None, :],
        other=0.0,
    ).to(tl.float32)

    # 加载 layer_out
    l_vals = tl.load(
        l_base + offset_channel * stride_l_c, mask=mask_channel, other=0.0
    ).to(tl.float32)

    # -----------------------------------------------------------
    # Step 4: 计算并立即写回 Grad X
    # -----------------------------------------------------------
    grad_x_acc = tl.zeros([NSIZE, BLOCK_CHANNEL], dtype=tl.float32)

    for k in tl.static_range(NSIZE):
        # 1. 加载 H 的第 k 行
        H_row_k_ptr = H_res_base + k * stride_H_n1 + index_n * stride_H_n2
        H_row_k_val = tl.load(H_row_k_ptr).to(tl.float32)
        H_row_k_val = H_row_k_val[:, None]  # [N, 1]

        # 2. 加载 g_out 的第 k 行
        g_out_row_k_ptr = (
            g_out_base + k * stride_g_out_n + offset_channel * stride_g_out_c
        )
        g_out_row_k = tl.load(g_out_row_k_ptr, mask=mask_channel, other=0.0).to(
            tl.float32
        )
        g_out_row_k = g_out_row_k[None, :]  # [1, BLOCK_C]

        grad_x_acc += H_row_k_val * g_out_row_k

    tl.store(
        g_x_base
        + index_n[:, None] * stride_grad_x_n
        + offset_channel[None, :] * stride_grad_x_c,
        grad_x_acc.to(tl.bfloat16),
        mask=mask_channel[None, :],
    )

    # -----------------------------------------------------------
    # Step 5: 计算并立即写回 Grad Layer Out
    # -----------------------------------------------------------
    grad_l_vals = tl.sum(g_out_vals * weight_vals[:, None], axis=0)
    tl.store(
        g_l_base + offset_channel * stride_grad_l_c,
        grad_l_vals.to(tl.bfloat16),
        mask=mask_channel,
    )

    # -----------------------------------------------------------
    # Step 6: 计算 Grad h (向量归约)
    # -----------------------------------------------------------
    grad_W_local = tl.sum(g_out_vals * l_vals[None, :], axis=1)
    grad_h_final = grad_W_local * dsig_h

    if BLOCK_CHANNEL >= CHANNEL_SIZE:
        tl.store(g_h_post_base + index_n * stride_grad_h_n, grad_h_final)
    else:
        tl.atomic_add(g_h_post_base + index_n * stride_grad_h_n, grad_h_final)

    # -----------------------------------------------------------
    # Step 7: 计算 Grad H (矩阵归约 - 按行向量化写入)
    # -----------------------------------------------------------
    # dH = g_out @ x^T
    # 我们需要 x 全量数据 (N, BLOCK_C)
    x_vals = tl.load(
        x_base + index_n[:, None] * stride_x_n + offset_channel[None, :] * stride_x_c,
        mask=mask_channel[None, :],
        other=0.0,
    ).to(tl.float32)

    # 按行计算 Grad H
    # dH 的第 i 行 = g_out[i, :] * x_vals.T
    #              = sum(g_out[i, :][None, :] * x_vals, axis=1)  <-- Broadcasting Magic

    for i in tl.static_range(NSIZE):
        # 1. 加载 g_out 的第 i 行: [1, BLOCK_C]
        # 这里虽然在循环里，但重复利用了 g_out_base，L1 Cache 会极快
        # (其实也可以直接用 g_out_vals[i][None, :] 但为了避开切片bug，我们重读)
        g_out_row_i_ptr = (
            g_out_base + i * stride_g_out_n + offset_channel * stride_g_out_c
        )
        g_out_row_i = tl.load(g_out_row_i_ptr, mask=mask_channel, other=0.0).to(
            tl.float32
        )

        # 2. 计算第 i 行的所有 N 个元素
        # g_out_row_i: [BLOCK_C]
        # x_vals:      [N, BLOCK_C]
        # 广播乘法: [1, BLOCK_C] * [N, BLOCK_C] -> [N, BLOCK_C]
        # 然后沿着 Channel 维度求和 -> [N]
        # 结果就是 dH[i, 0], dH[i, 1] ... dH[i, N-1]
        grad_H_row_i = tl.sum(g_out_row_i[None, :] * x_vals, axis=1)  # Shape: [N]

        # 3. 写入第 i 行
        # 计算 dH 中第 i 行的起始地址
        off_H_row_start = i * stride_grad_H_n1 + index_n * stride_grad_H_n2
        target_ptrs = g_H_res_base + off_H_row_start

        if BLOCK_CHANNEL >= CHANNEL_SIZE:
            tl.store(target_ptrs, grad_H_row_i)
        else:
            tl.atomic_add(target_ptrs, grad_H_row_i)
