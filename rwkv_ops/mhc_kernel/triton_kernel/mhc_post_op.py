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
    # [新] Chunk Strides: 用于控制写入 Workspace 的偏移
    # 如果 Grid_Y=1，这些 stride 为 0；如果 Grid_Y>1，这些 stride 为对应 Tensor 的大小
    stride_grad_h_bt: tl.constexpr,
    stride_grad_h_chunk: tl.constexpr,
    stride_grad_h_n: tl.constexpr,
    stride_grad_H_bt: tl.constexpr,
    stride_grad_H_chunk: tl.constexpr,
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

    # [关键] 归约梯度的地址计算
    # 地址 = Base + (Time_Offset) + (Chunk_Offset) + (Elem_Offset)
    # 如果只有1个块，Chunk_Offset 为 0，直接写回原位。
    # 如果有多个块，每个块写入自己独立的 Workspace 区域。
    g_h_post_base = (
        grad_h_ptr
        + pid_batch_time * stride_grad_h_bt
        + pid_channel_block * stride_grad_h_chunk
    )

    g_H_res_base = (
        grad_H_ptr
        + pid_batch_time * stride_grad_H_bt
        + pid_channel_block * stride_grad_H_chunk
    )

    # -----------------------------------------------------------
    # Step 2: 预加载 Vector h
    # -----------------------------------------------------------
    index_n = tl.arange(0, NSIZE)
    h_vals = tl.load(h_post_base + index_n * stride_h_n).to(tl.float32)
    sig_h = tl.sigmoid(h_vals)
    weight_vals = sig_h * 2.0
    dsig_h = sig_h * (1.0 - sig_h) * 2.0

    # -----------------------------------------------------------
    # Step 3: 加载流式数据块
    # -----------------------------------------------------------
    g_out_vals = tl.load(
        g_out_base
        + index_n[:, None] * stride_g_out_n
        + offset_channel[None, :] * stride_g_out_c,
        mask=mask_channel[None, :],
        other=0.0,
    ).to(tl.float32)

    l_vals = tl.load(
        l_base + offset_channel * stride_l_c, mask=mask_channel, other=0.0
    ).to(tl.float32)

    # -----------------------------------------------------------
    # Step 4: 计算并立即写回 Grad X
    # -----------------------------------------------------------
    grad_x_acc = tl.zeros([NSIZE, BLOCK_CHANNEL], dtype=tl.float32)
    for k in tl.static_range(NSIZE):
        # 1. 加载 H 行
        H_row_k_ptr = H_res_base + k * stride_H_n1 + index_n * stride_H_n2
        H_row_k_val = tl.load(H_row_k_ptr).to(tl.float32)
        H_row_k_val = H_row_k_val[:, None]

        # 2. 加载 g_out 行
        g_out_row_k_ptr = (
            g_out_base + k * stride_g_out_n + offset_channel * stride_g_out_c
        )
        g_out_row_k = tl.load(g_out_row_k_ptr, mask=mask_channel, other=0.0).to(
            tl.float32
        )
        g_out_row_k = g_out_row_k[None, :]

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
    # Step 6: 计算 Grad h 并写入 (Store Only)
    # -----------------------------------------------------------
    grad_W_local = tl.sum(g_out_vals * l_vals[None, :], axis=1)
    grad_h_final = grad_W_local * dsig_h

    # 直接写入 (地址已包含 Chunk Offset)
    tl.store(g_h_post_base + index_n * stride_grad_h_n, grad_h_final)
    x_vals = tl.load(
        x_base + index_n[:, None] * stride_x_n + offset_channel[None, :] * stride_x_c,
        mask=mask_channel[None, :],
        other=0.0,
    ).to(tl.float32)

    for i in tl.static_range(NSIZE):
        g_out_row_i_ptr = (
            g_out_base + i * stride_g_out_n + offset_channel * stride_g_out_c
        )
        g_out_row_i = tl.load(g_out_row_i_ptr, mask=mask_channel, other=0.0).to(
            tl.float32
        )

        grad_H_row_i = tl.sum(g_out_row_i[None, :] * x_vals, axis=1)  # Shape: [N]
        off_H_row_start = i * stride_grad_H_n1 + index_n * stride_grad_H_n2
        target_ptrs = g_H_res_base + off_H_row_start

  
        tl.store(target_ptrs, grad_H_row_i)
