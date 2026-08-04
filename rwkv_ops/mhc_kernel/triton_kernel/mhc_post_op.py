"""mHC Post-Op 共享 Triton 内核（Distribute + Mix 融合）。"""

import triton
import triton.language as tl


# mHC Post-Op 前向 Triton kernel。
#
# 每个 program 处理一个 (batch*time, channel_block) tile，沿 n 维度静态展开，
# 将 layer_out 经 h_post 门控广播后，与 H_res @ x_expanded 累加得到输出。
#
# Args:
#   x_expanded_ptr: [Total_BT, n, C], bfloat16, row-major。原始多流输入。
#   h_post_raw_ptr: [Total_BT, n], float32, row-major。未激活分发权重。
#   H_res_ptr: [Total_BT, n, n], float32, row-major。双随机流混合矩阵。
#   layer_out_ptr: [Total_BT, C], bfloat16, row-major。核心层输出。
#   output_ptr: [Total_BT, n, C], bfloat16, row-major。融合输出。
#
# 编译期宏:
#   CHANNEL_SIZE, NSIZE, BLOCK_CHANNEL。
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_CHANNEL": block_c}, num_warps=num_warps, num_stages=num_stages
        )
        for block_c in [128, 256, 512, 1024]
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["CHANNEL_SIZE"],
)
@triton.jit
def mhc_fused_forward_kernel(
    x_expanded_ptr,
    h_post_raw_ptr,
    H_res_ptr,
    layer_out_ptr,
    output_ptr,
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
    CHANNEL_SIZE: tl.constexpr,
    NSIZE: tl.constexpr,
    BLOCK_CHANNEL: tl.constexpr,
):
    pid_batch_time = tl.program_id(0)
    pid_channel_block = tl.program_id(1)

    offset_channel = pid_channel_block * BLOCK_CHANNEL + tl.arange(0, BLOCK_CHANNEL)
    mask_channel = offset_channel < CHANNEL_SIZE

    x_expanded_base = x_expanded_ptr + pid_batch_time * stride_x_batch_time
    h_post_raw_base = h_post_raw_ptr + pid_batch_time * stride_h_batch_time
    H_res_base = H_res_ptr + pid_batch_time * stride_H_batch_time
    layer_out_base = layer_out_ptr + pid_batch_time * stride_layer_out_batch_time
    output_base = output_ptr + pid_batch_time * stride_output_batch_time

    index_n = tl.arange(0, NSIZE)

    h_post_values = tl.load(h_post_raw_base + index_n * stride_h_n_size).to(tl.float32)
    weight_values = tl.sigmoid(h_post_values) * 2.0

    layer_out_values = tl.load(
        layer_out_base + offset_channel * stride_layer_out_channel,
        mask=mask_channel,
        other=0.0,
    ).to(tl.float32)

    accumulator = weight_values[:, None] * layer_out_values[None, :]

    for k in tl.static_range(NSIZE):
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


# mHC Post-Op 反向 Triton kernel。
#
# 每个 program 处理一个 batch*time 实例，沿 Channel 维度循环分块。
# 四个梯度（grad_x, grad_h, grad_H, grad_layer_out）在一个 kernel 内融合计算，
# grad_h 与 grad_H 在 persistent 寄存器中累加，避免 atomic_add。
#
# Args:
#   x_ptr, h_ptr, H_ptr, l_ptr, g_ptr: 前向输入与上游梯度，形状同前向。
#   gx_ptr, gh_ptr, gH_ptr, gl_ptr: 输出梯度。
#
# 编译期宏:
#   CHANNEL_SIZE, NSIZE, BLOCK_CHANNEL。
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_CHANNEL": block_c}, num_warps=num_warps, num_stages=num_stages
        )
        for block_c in [128, 256, 512, 1024]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["CHANNEL_SIZE"],
)
@triton.jit
def mhc_fused_backward_kernel(
    x_ptr,
    h_ptr,
    H_ptr,
    l_ptr,
    g_ptr,
    gx_ptr,
    gh_ptr,
    gH_ptr,
    gl_ptr,
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
    CHANNEL_SIZE: tl.constexpr,
    NSIZE: tl.constexpr,
    BLOCK_CHANNEL: tl.constexpr,
):
    pid_bt = tl.program_id(0)

    off_n = tl.arange(0, NSIZE)
    mask_H = (off_n[:, None] < NSIZE) & (off_n[None, :] < NSIZE)

    p_h = h_ptr + pid_bt * stride_h_bt + off_n * stride_h_n
    p_H = (
        H_ptr
        + pid_bt * stride_H_bt
        + (off_n[:, None] * stride_H_n1 + off_n[None, :] * stride_H_n2)
    )

    # h 与 H 在整个 C 循环中为常数，预加载到寄存器减少重复访存。
    h_vals = tl.load(p_h).to(tl.float32)
    sig_h = tl.sigmoid(h_vals)
    w_vals = sig_h * 2.0
    dw_vals = sig_h * (1.0 - sig_h) * 2.0

    H_vals = tl.load(p_H, mask=mask_H, other=0.0).to(tl.float32)

    acc_gh = tl.zeros([NSIZE], dtype=tl.float32)
    acc_gH = tl.zeros([NSIZE, NSIZE], dtype=tl.float32)

    for start_c in tl.static_range(0, CHANNEL_SIZE, BLOCK_CHANNEL):
        off_c = start_c + tl.arange(0, BLOCK_CHANNEL)
        mask_c = off_c < CHANNEL_SIZE
        mask_2d = (off_n[:, None] < NSIZE) & (off_c[None, :] < CHANNEL_SIZE)

        l_chunk = tl.load(
            l_ptr + pid_bt * stride_l_bt + off_c * stride_l_c, mask=mask_c, other=0.0
        ).to(tl.float32)
        x_chunk = tl.load(
            x_ptr
            + pid_bt * stride_x_bt
            + (off_n[:, None] * stride_x_n + off_c[None, :] * stride_x_c),
            mask=mask_2d,
            other=0.0,
        ).to(tl.float32)
        g_chunk = tl.load(
            g_ptr
            + pid_bt * stride_g_bt
            + (off_n[:, None] * stride_g_n + off_c[None, :] * stride_g_c),
            mask=mask_2d,
            other=0.0,
        ).to(tl.float32)

        gl_chunk = tl.sum(g_chunk * w_vals[:, None], axis=0)
        tl.store(
            gl_ptr + pid_bt * stride_gl_bt + off_c * stride_gl_c,
            gl_chunk.to(tl.bfloat16),
            mask=mask_c,
        )

        # gx = H^T @ g，通过广播累加实现。
        gx_chunk = tl.sum(H_vals[:, :, None] * g_chunk[:, None, :], axis=0)
        tl.store(
            gx_ptr
            + pid_bt * stride_gx_bt
            + (off_n[:, None] * stride_gx_n + off_c[None, :] * stride_gx_c),
            gx_chunk.to(tl.bfloat16),
            mask=mask_2d,
        )

        acc_gh += tl.sum(g_chunk * l_chunk[None, :], axis=1) * dw_vals
        acc_gH += tl.sum(g_chunk[:, None, :] * x_chunk[None, :, :], axis=2)

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
