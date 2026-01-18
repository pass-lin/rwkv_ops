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

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_CHANNEL": block_c}, num_warps=num_warps, num_stages=num_stages
        )
        for block_c in [128, 256, 512, 1024, 2048]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["CHANNEL_SIZE"],
)
@triton.jit
def mhc_fused_backward_kernel_workspace(
    # --- 指针 ---
    x_ptr, h_ptr, H_ptr, l_ptr, g_ptr,
    gx_ptr, gh_ws_ptr, gH_ws_ptr, gl_ptr,
    # --- 步幅 ---
    stride_x_bt, stride_x_n, stride_x_c,
    stride_h_bt, stride_h_n,
    stride_H_bt, stride_H_n1, stride_H_n2,
    stride_l_bt, stride_l_c,
    stride_g_bt, stride_g_n, stride_g_c,
    stride_gx_bt, stride_gx_n, stride_gx_c,
    stride_gl_bt, stride_gl_c,
    stride_gh_bt, stride_gh_chunk, stride_gh_n,
    stride_gH_bt, stride_gH_chunk, stride_gH_n1, stride_gH_n2,
    # --- 常量 ---
    CHANNEL_SIZE: tl.constexpr,
    NSIZE: tl.constexpr,
    BLOCK_CHANNEL: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_c = tl.program_id(1)

    # 1. 构造索引
    off_n = tl.arange(0, NSIZE)
    off_c = pid_c * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    
    # Mask
    mask_c = off_c < CHANNEL_SIZE
    mask_2d = (off_n[:, None] < NSIZE) & (off_c[None, :] < CHANNEL_SIZE)
    mask_H = (off_n[:, None] < NSIZE) & (off_n[None, :] < NSIZE)

    # 2. 指针计算 (与输入 Tensor 对应)
    # 输入
    p_h = h_ptr + pid_bt * stride_h_bt + off_n * stride_h_n
    p_H = H_ptr + pid_bt * stride_H_bt + (off_n[:, None] * stride_H_n1 + off_n[None, :] * stride_H_n2)
    p_l = l_ptr + pid_bt * stride_l_bt + off_c * stride_l_c
    p_x = x_ptr + pid_bt * stride_x_bt + (off_n[:, None] * stride_x_n + off_c[None, :] * stride_x_c)
    p_g = g_ptr + pid_bt * stride_g_bt + (off_n[:, None] * stride_g_n + off_c[None, :] * stride_g_c)

    # 输出 (Workspace)
    p_gh_ws = gh_ws_ptr + pid_bt * stride_gh_bt + pid_c * stride_gh_chunk + off_n * stride_gh_n
    p_gH_ws = gH_ws_ptr + pid_bt * stride_gH_bt + pid_c * stride_gH_chunk + \
              (off_n[:, None] * stride_gH_n1 + off_n[None, :] * stride_gH_n2)

    # 3. 加载到寄存器 (Load once)
    h_vals = tl.load(p_h).to(tl.float32)  # [N]
    sig_h = tl.sigmoid(h_vals)
    w_vals = sig_h * 2.0
    dw_vals = sig_h * (1.0 - sig_h) * 2.0

    H_vals = tl.load(p_H, mask=mask_H, other=0.0).to(tl.float32)  # [N, N]
    l_chunk = tl.load(p_l, mask=mask_c, other=0.0).to(tl.float32) # [C]
    x_chunk = tl.load(p_x, mask=mask_2d, other=0.0).to(tl.float32) # [N, C]
    g_chunk = tl.load(p_g, mask=mask_2d, other=0.0).to(tl.float32) # [N, C]

    # -----------------------------------------------------------
    # 4. 计算逻辑 (无循环，全广播)
    # -----------------------------------------------------------

    # Task A: grad_layer_out [C]
    # gl = sum_n (g_out[n, c] * w[n])
    gl_acc = tl.sum(g_chunk * w_vals[:, None], axis=0)
    p_gl = gl_ptr + pid_bt * stride_gl_bt + off_c * stride_gl_c
    tl.store(p_gl, gl_acc.to(tl.bfloat16), mask=mask_c)

    # Task B: grad_x [N, C]
    # dx = H^T @ g_out -> dx[j, c] = sum_i (H[i, j] * g_out[i, c])
    # H[i, j, 1] * g[i, 1, c] -> [i, j, c] -> sum over i (axis 0)
    gx_acc = tl.sum(H_vals[:, :, None] * g_chunk[:, None, :], axis=0)
    p_gx = gx_ptr + pid_bt * stride_gx_bt + (off_n[:, None] * stride_gx_n + off_c[None, :] * stride_gx_c)
    tl.store(p_gx, gx_acc.to(tl.bfloat16), mask=mask_2d)

    # Task C: grad_h [N] (写入 Workspace)
    # dh = sum_c (g[n, c] * l[c]) * dw[n]
    gh_acc = tl.sum(g_chunk * l_chunk[None, :], axis=1) * dw_vals
    tl.store(p_gh_ws, gh_acc.to(tl.float32))

    # Task D: grad_H [N, N] (写入 Workspace)
    # dH[i, j] = sum_c (g[i, c] * x[j, c])
    # g[i, 1, c] * x[1, j, c] -> [i, j, c] -> sum over c (axis 2)
    gH_acc = tl.sum(g_chunk[:, None, :] * x_chunk[None, :, :], axis=2)
    tl.store(p_gH_ws, gH_acc.to(tl.float32), mask=mask_H)