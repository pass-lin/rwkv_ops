import jax
import jax.numpy as jnp
import jax_triton as jt
from ..triton_kernel.mhc_post_op import *


def mhc_post_op_fwd_kernel_call(
    layer_out: jax.Array, x_expanded: jax.Array, h_post_raw: jax.Array, H_res: jax.Array
):
    batch, time, NSIZE, channel = x_expanded.shape
    total_bt = batch * time

    # 1. 重塑形状 (JAX 中不需要像 Torch 那样显式调用 .contiguous())
    x_v = x_expanded.reshape(total_bt, NSIZE, channel)
    h_v = h_post_raw.reshape(total_bt, NSIZE)
    H_v = H_res.reshape(total_bt, NSIZE, NSIZE)
    l_v = layer_out.reshape(total_bt, channel)

    # 2. 计算步长
    sx_bt, sx_n, sx_c = jt.strides_from_shape(x_v.shape)
    sh_bt, sh_n = jt.strides_from_shape(h_v.shape)
    sH_bt, sH_n1, sH_n2 = jt.strides_from_shape(H_v.shape)
    sl_bt, sl_c = jt.strides_from_shape(l_v.shape)

    # 输出形状与 x_v 一致
    out_struct = jax.ShapeDtypeStruct(x_v.shape, x_v.dtype)
    so_bt, so_n, so_c = sx_bt, sx_n, sx_c

    # 3. 定义 Grid
    # 注意：BLOCK_CHANNEL 需要在调用时作为元参数传入
    grid = lambda meta: (total_bt, jt.cdiv(channel, meta["BLOCK_CHANNEL"]))

    # 4. 调用 Triton
    out_v = jt.triton_call(
        x_v,
        h_v,
        H_v,
        l_v,  # 输入 Array
        kernel=mhc_fused_forward_kernel,
        out_shape=out_struct,  # 输出 Array (由 JAX 自动追加在 l_v 之后)
        grid=grid,
        # 标量/步长参数 (作为 kwargs 传入，对应 Kernel 中的参数名)
        stride_output_batch_time=so_bt,
        stride_output_n_size=so_n,
        stride_output_channel=so_c,
        stride_x_batch_time=sx_bt,
        stride_x_n_size=sx_n,
        stride_x_channel=sx_c,
        stride_h_batch_time=sh_bt,
        stride_h_n_size=sh_n,
        stride_H_batch_time=sH_bt,
        stride_H_n_size_1=sH_n1,
        stride_H_n_size_2=sH_n2,
        stride_layer_out_batch_time=sl_bt,
        stride_layer_out_channel=sl_c,
        CHANNEL_SIZE=channel,
        NSIZE=NSIZE,
    )

    return out_v.reshape(batch, time, NSIZE, channel)


def mhc_post_op_bwd_kernel_call(res, grad_output):
    layer_out, x_expanded, h_post_raw, H_res = res
    B, T, n, C = x_expanded.shape
    total_bt = B * T

    # 1. 准备输入
    x_v = x_expanded.reshape(total_bt, n, C)
    h_v = h_post_raw.reshape(total_bt, n)
    H_v = H_res.reshape(total_bt, n, n)
    l_v = layer_out.reshape(total_bt, C)
    g_out_v = grad_output.reshape(total_bt, n, C)

    # 2. 计算输入步长
    sx_bt, sx_n, sx_c = jt.strides_from_shape(x_v.shape)
    sh_bt, sh_n = jt.strides_from_shape(h_v.shape)
    sH_bt, sH_n1, sH_n2 = jt.strides_from_shape(H_v.shape)
    sl_bt, sl_c = jt.strides_from_shape(l_v.shape)
    sg_bt, sg_n, sg_c = jt.strides_from_shape(g_out_v.shape)

    # 3. 准备输出结构 (grad_x, grad_h, grad_H, grad_l)
    # 根据原代码，h 和 H 的梯度使用 float32
    out_shapes = [
        jax.ShapeDtypeStruct(x_v.shape, x_v.dtype),  # grad_x
        jax.ShapeDtypeStruct(h_v.shape, jnp.float32),  # grad_h
        jax.ShapeDtypeStruct(H_v.shape, jnp.float32),  # grad_H
        jax.ShapeDtypeStruct(l_v.shape, l_v.dtype),  # grad_l
    ]

    # 4. 计算输出步长 (用于传给 Kernel)
    sgx_bt, sgx_n, sgx_c = sx_bt, sx_n, sx_c
    sgh_bt, sgh_n = sh_bt, sh_n
    sgH_bt, sgH_n1, sgH_n2 = sH_bt, sH_n1, sH_n2
    sgl_bt, sgl_c = sl_bt, sl_c

    # 5. 调用 Triton
    # 注意：positional args 为输入，out_shape 对应的输出会自动追加在后
    grad_x_v, grad_h_v, grad_H_v, grad_l_v = jt.triton_call(
        x_v,
        h_v,
        H_v,
        l_v,
        g_out_v,
        kernel=mhc_fused_backward_kernel,
        out_shape=out_shapes,
        grid=(total_bt, 1),
        # 对应 Kernel 参数名
        stride_x_bt=sx_bt,
        stride_x_n=sx_n,
        stride_x_c=sx_c,
        stride_h_bt=sh_bt,
        stride_h_n=sh_n,
        stride_H_bt=sH_bt,
        stride_H_n1=sH_n1,
        stride_H_n2=sH_n2,
        stride_l_bt=sl_bt,
        stride_l_c=sl_c,
        stride_g_bt=sg_bt,
        stride_g_n=sg_n,
        stride_g_c=sg_c,
        stride_gx_bt=sgx_bt,
        stride_gx_n=sgx_n,
        stride_gx_c=sgx_c,
        stride_gl_bt=sgl_bt,
        stride_gl_c=sgl_c,
        stride_gh_bt=sgh_bt,
        stride_gh_n=sgh_n,
        stride_gH_bt=sgH_bt,
        stride_gH_n1=sgH_n1,
        stride_gH_n2=sgH_n2,
        CHANNEL_SIZE=C,
        NSIZE=n,
    )

    # 6. 返回梯度 (需与 mhc_post_op 的输入顺序一致)
    return (
        grad_l_v.reshape(B, T, C),
        grad_x_v.reshape(B, T, n, C),
        grad_h_v.reshape(B, T, n).astype(h_post_raw.dtype),
        grad_H_v.reshape(B, T, n, n).astype(H_res.dtype),
    )


# --- JAX 接口绑定 ---


@jax.custom_vjp
def mhc_post_op(layer_out, x_expanded, h_post_raw, H_res):
    # 这里的逻辑对应原始接口的类型转换
    return mhc_post_op_fwd_kernel_call(
        layer_out.astype(jnp.bfloat16),
        x_expanded.astype(jnp.bfloat16),
        h_post_raw.astype(jnp.float32),
        H_res.astype(jnp.float32),
    )


def mhc_post_op_fwd(layer_out, x_expanded, h_post_raw, H_res):
    # 执行前向并保存 residual
    out = mhc_post_op(layer_out, x_expanded, h_post_raw, H_res)
    return out, (layer_out, x_expanded, h_post_raw, H_res)


def mhc_post_op_bwd(res, grad_output):
    # 调用反向 Kernel
    grads = mhc_post_op_bwd_kernel_call(res, grad_output)
    return grads


mhc_post_op.defvjp(mhc_post_op_fwd, mhc_post_op_bwd)
