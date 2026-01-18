from ..triton_kernel.mhc_pre_op import *
import jax
import jax.numpy as jnp
import jax_triton as jt
from functools import partial


# --- 1. 前向 Launcher ---
def mhc_pre_op_fwd_kernel_call(x, h_res_in, h_pre_in, num_iters, eps):
    B, T, n, C = x.shape
    total_bt = B * T

    # 重塑视图
    x_v = x.reshape(total_bt, n, C)
    h_res_v = h_res_in.reshape(total_bt, n, n)
    h_pre_v = h_pre_in.reshape(total_bt, n)

    # 计算输入步长
    sx_bt, sx_n, sx_c = jt.strides_from_shape(x_v.shape)
    shr_bt, shr_n1, shr_n2 = jt.strides_from_shape(h_res_v.shape)
    shp_bt, shp_n = jt.strides_from_shape(h_pre_v.shape)

    # 准备输出结构 (注意：out 是 BF16, H_res_out 是 FP32)
    out_shapes = [
        jax.ShapeDtypeStruct((total_bt, C), x.dtype),  # out_ptr
        jax.ShapeDtypeStruct((total_bt, n, n), jnp.float32),  # H_res_out_ptr
    ]

    # 计算输出步长
    so_bt, so_c = jt.strides_from_shape(out_shapes[0].shape)
    sHr_bt, sHr_n1, sHr_n2 = jt.strides_from_shape(out_shapes[1].shape)

    # 定义 Grid (匹配你代码里的 lambda)
    grid = lambda meta: (
        jt.cdiv(total_bt, meta["BLOCK_BT"]),
        jt.cdiv(C, meta["BLOCK_C"]),
    )

    # 调用 Triton: 这里的关键字参数必须与你 Forward Kernel 的参数名完全一致
    out_v, H_res_out_v = jt.triton_call(
        x_v,
        h_res_v,
        h_pre_v,  # 指针参数 1, 2, 3
        kernel=sinkhorn_aggregate_fused_kernel,
        out_shape=out_shapes,  # 产生指针参数 4, 5
        grid=grid,
        # --- 标量参数: 必须匹配前向 Kernel 的定义 ---
        Total_BT_CONST=total_bt,
        NSIZE=n,
        CSIZE=C,
        NUM_ITERS=num_iters,
        EPS=eps,
        stride_x_bt=sx_bt,
        stride_x_n=sx_n,
        stride_x_c=sx_c,
        stride_h_res_in_bt=shr_bt,
        stride_h_res_in_n1=shr_n1,
        stride_h_res_in_n2=shr_n2,
        stride_h_pre_in_bt=shp_bt,
        stride_h_pre_in_n=shp_n,
        stride_out_bt=so_bt,
        stride_out_c=so_c,
        stride_Hr_out_bt=sHr_bt,
        stride_Hr_out_n1=sHr_n1,
        stride_Hr_out_n2=sHr_n2,
    )

    return out_v.reshape(B, T, C), H_res_out_v.reshape(B, T, n, n)


# --- 2. 反向 Launcher ---
def mhc_pre_op_bwd_kernel_call(x, h_res, h_pre, grad_out, grad_H_res, num_iters, eps):
    B, T, n, C = x.shape
    total_bt = B * T

    # 视图重塑
    x_v = x.reshape(total_bt, n, C)
    h_v = h_res.reshape(total_bt, n, n)
    h_p_v = h_pre.reshape(total_bt, n)
    g_out_v = grad_out.reshape(total_bt, C)
    g_H_v = grad_H_res.reshape(total_bt, n, n)

    # 步幅
    sgout_bt, sgout_c = jt.strides_from_shape(g_out_v.shape)
    sgH_bt, sgH_n1, sgH_n2 = jt.strides_from_shape(g_H_v.shape)
    sx_bt, sx_n, sx_c = jt.strides_from_shape(x_v.shape)
    sh_bt, sh_n1, sh_n2 = jt.strides_from_shape(h_v.shape)
    shp_bt, shp_n = jt.strides_from_shape(h_p_v.shape)

    # 准备输出梯度结构 (gx, gh_res, gh_pre)
    out_shapes = [
        jax.ShapeDtypeStruct(x_v.shape, x.dtype),
        jax.ShapeDtypeStruct(h_v.shape, h_res.dtype),
        jax.ShapeDtypeStruct(h_p_v.shape, h_pre.dtype),
    ]

    # 获取输出步幅传给反向 kernel
    sgx_bt, sgx_n, sgx_c = sx_bt, sx_n, sx_c
    sghr_bt, sghr_n1, sghr_n2 = sh_bt, sh_n1, sh_n2
    sghp_bt, sghp_n = shp_bt, shp_n

    # 调用 Triton: 这里的关键字参数必须与你 Backward Kernel 的参数名完全一致
    gx_v, gh_res_v, gh_pre_v = jt.triton_call(
        g_out_v,
        g_H_v,
        x_v,
        h_v,
        h_p_v,  # 输入指针 1, 2, 3, 4, 5
        kernel=sinkhorn_aggregate_bwd_kernel,
        out_shape=out_shapes,  # 产生输出指针 6, 7, 8
        grid=(total_bt, 1),
        # --- 标量参数: 必须匹配反向 Kernel 的定义 ---
        TOTAL_BT_CONST=total_bt,
        NSIZE=n,
        CHANNEL_SIZE=C,
        NUM_ITERS=num_iters,
        EPS=eps,
        stride_gout_bt=sgout_bt,
        stride_gout_c=sgout_c,
        stride_gH_bt=sgH_bt,
        stride_gH_n1=sgH_n1,
        stride_gH_n2=sgH_n2,
        stride_x_bt=sx_bt,
        stride_x_n=sx_n,
        stride_x_c=sx_c,
        stride_h_res_bt=sh_bt,
        stride_h_res_n1=sh_n1,
        stride_h_res_n2=sh_n2,
        stride_h_pre_bt=shp_bt,
        stride_h_pre_n=shp_n,
        stride_gx_bt=sgx_bt,
        stride_gx_n=sgx_n,
        stride_gx_c=sgx_c,
        stride_gh_res_bt=sghr_bt,
        stride_gh_res_n1=sghr_n1,
        stride_gh_res_n2=sghr_n2,
        stride_gh_pre_bt=sghp_bt,
        stride_gh_pre_n=sghp_n,
    )

    return (
        gx_v.reshape(B, T, n, C),
        gh_res_v.reshape(B, T, n, n),
        gh_pre_v.reshape(B, T, n),
    )


# --- 3. JAX 接口绑定 (解决 Tracer Leak) ---


@partial(jax.jit, static_argnums=(3, 4))
def mhc_pre_op_fused(x, h_res_in, h_pre_in, num_iters=20, eps=1e-8):
    """
    通过闭包捕获静态参数 num_iters 和 eps，确保它们不进入 custom_vjp 的追踪范围。
    """

    @jax.custom_vjp
    def _internal_op(x_arr, hr_arr, hp_arr):
        return mhc_pre_op_fwd_kernel_call(x_arr, hr_arr, hp_arr, num_iters, eps)

    def _internal_fwd(x_arr, hr_arr, hp_arr):
        out_tuple = _internal_op(x_arr, hr_arr, hp_arr)
        # 只保存参与微分的张量
        return out_tuple, (x_arr, hr_arr, hp_arr)

    def _internal_bwd(res, grads):
        x_arr, hr_arr, hp_arr = res
        grad_out, grad_H_res = grads

        # 处理可能的 None 梯度（虽然 JAX 通常会传零 Tensor，但为了健壮性）
        if grad_out is None:
            grad_out = jnp.zeros_like(x_arr[:, :, 0])
        if grad_H_res is None:
            grad_H_res = jnp.zeros_like(hr_arr)

        gx, ghr, ghp = mhc_pre_op_bwd_kernel_call(
            x_arr, hr_arr, hp_arr, grad_out, grad_H_res, num_iters, eps
        )
        return gx, ghr, ghp

    _internal_op.defvjp(_internal_fwd, _internal_bwd)

    return _internal_op(x, h_res_in, h_pre_in)
