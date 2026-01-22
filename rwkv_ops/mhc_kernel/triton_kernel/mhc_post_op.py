import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_CHANNEL": block_c}, num_warps=num_warps, num_stages=num_stages
        )
        for block_c in [128, 256, 512,1024]
        for num_warps in [ 4, 8]
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
        for block_c in [128, 256, 512,1024]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["CHANNEL_SIZE"],
)
@triton.jit
def mhc_fused_backward_kernel(
    # --- 指针 ---
    x_ptr,
    h_ptr,
    H_ptr,
    l_ptr,
    g_ptr,
    gx_ptr,
    gh_ptr,
    gH_ptr,
    gl_ptr,
    # --- 步幅 ---
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
    stride_g_bt: tl.constexpr,
    stride_g_n: tl.constexpr,
    stride_g_c: tl.constexpr,
    stride_gx_bt: tl.constexpr,
    stride_gx_n: tl.constexpr,
    stride_gx_c: tl.constexpr,
    stride_gl_bt: tl.constexpr,
    stride_gl_c: tl.constexpr,
    stride_gh_bt: tl.constexpr,
    stride_gh_n: tl.constexpr,
    stride_gH_bt: tl.constexpr,
    stride_gH_n1: tl.constexpr,
    stride_gH_n2: tl.constexpr,
    # --- 常量 ---
    CHANNEL_SIZE: tl.constexpr,
    NSIZE: tl.constexpr,
    BLOCK_CHANNEL: tl.constexpr,
):
    # 1. 获取 Program ID (仅处理 Batch * Time 维度)
    pid_bt = tl.program_id(0)

    # 2. 构造 N 维度的索引 (静态长度)
    off_n = tl.arange(0, NSIZE)
    mask_H = (off_n[:, None] < NSIZE) & (off_n[None, :] < NSIZE)

    # 3. 基础指针偏移
    p_h = h_ptr + pid_bt * stride_h_bt + off_n * stride_h_n
    p_H = (
        H_ptr
        + pid_bt * stride_H_bt
        + (off_n[:, None] * stride_H_n1 + off_n[None, :] * stride_H_n2)
    )

    # 4. 预加载 H 矩阵 [N, N] 和 h 激活值 [N]
    # 这些在处理整行 C 的过程中是常数，存在寄存器里
    h_vals = tl.load(p_h).to(tl.float32)
    sig_h = tl.sigmoid(h_vals)
    w_vals = sig_h * 2.0
    dw_vals = sig_h * (1.0 - sig_h) * 2.0

    H_vals = tl.load(p_H, mask=mask_H, other=0.0).to(tl.float32)

    # 5. 初始化规约累加器 (Persistent Registers)
    acc_gh = tl.zeros([NSIZE], dtype=tl.float32)
    acc_gH = tl.zeros([NSIZE, NSIZE], dtype=tl.float32)

    # -----------------------------------------------------------
    # 6. 主循环：跨步遍历 Channel 维度
    # -----------------------------------------------------------
    for start_c in tl.static_range(0, CHANNEL_SIZE, BLOCK_CHANNEL):
        off_c = start_c + tl.arange(0, BLOCK_CHANNEL)
        mask_c = off_c < CHANNEL_SIZE
        mask_2d = (off_n[:, None] < NSIZE) & (off_c[None, :] < CHANNEL_SIZE)

        # 加载分块数据
        # l_chunk: [C]
        l_chunk = tl.load(
            l_ptr + pid_bt * stride_l_bt + off_c * stride_l_c, mask=mask_c, other=0.0
        ).to(tl.float32)
        # x_chunk: [N, C]
        x_chunk = tl.load(
            x_ptr
            + pid_bt * stride_x_bt
            + (off_n[:, None] * stride_x_n + off_c[None, :] * stride_x_c),
            mask=mask_2d,
            other=0.0,
        ).to(tl.float32)
        # g_chunk: [N, C]
        g_chunk = tl.load(
            g_ptr
            + pid_bt * stride_g_bt
            + (off_n[:, None] * stride_g_n + off_c[None, :] * stride_g_c),
            mask=mask_2d,
            other=0.0,
        ).to(tl.float32)

        # --- Task A: grad_layer_out [C] ---
        # 结果 = sum_n (g[n, c] * w[n])
        gl_chunk = tl.sum(g_chunk * w_vals[:, None], axis=0)
        tl.store(
            gl_ptr + pid_bt * stride_gl_bt + off_c * stride_gl_c,
            gl_chunk.to(tl.bfloat16),
            mask=mask_c,
        )

        # --- Task B: grad_x [N, C] ---
        # 结果 = H^T @ g -> [N, N] @ [N, C]
        # 使用广播计算：H[i, j, 1] * g[i, 1, c] -> [i, j, c] -> sum over i
        gx_chunk = tl.sum(H_vals[:, :, None] * g_chunk[:, None, :], axis=0)
        tl.store(
            gx_ptr
            + pid_bt * stride_gx_bt
            + (off_n[:, None] * stride_gx_n + off_c[None, :] * stride_gx_c),
            gx_chunk.to(tl.bfloat16),
            mask=mask_2d,
        )

        # --- Task C: 累加 grad_h 分量 ---
        # 局部规约：sum_c (g[n, c] * l[c])
        acc_gh += tl.sum(g_chunk * l_chunk[None, :], axis=1) * dw_vals

        # --- Task D: 累加 grad_H 分量 ---
        # 局部规约：sum_c (g[i, c] * x[j, c])
        # 使用广播：g[i, 1, c] * x[1, j, c] -> [i, j, c] -> sum over c
        acc_gH += tl.sum(g_chunk[:, None, :] * x_chunk[None, :, :], axis=2)

    # -----------------------------------------------------------
    # 7. 循环结束：写回规约完成的 dh 和 dH
    # -----------------------------------------------------------
    tl.store(
        gh_ptr + pid_bt * stride_gh_bt + off_n * stride_gh_n, acc_gh.to(tl.float32)
    )
    tl.store(
        gH_ptr
        + pid_bt * stride_gH_bt
        + (off_n[:, None] * stride_gH_n1 + off_n[None, :] * stride_gH_n2),
        acc_gH.to(tl.float32),
        mask=mask_H,
    )
