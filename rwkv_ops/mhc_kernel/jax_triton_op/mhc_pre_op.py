from ..triton_kernel.mhc_pre_op import (
    sinkhorn_aggregate_bwd_kernel,
    sinkhorn_aggregate_fused_kernel,
)
import jax
import jax.numpy as jnp
import jax_triton as jt
from functools import partial
import jax.tree_util as jtu

# 引入自定义分区器和切分规则 (适配 JAX 新版 Shardy 引擎)
from jax.experimental.custom_partitioning import custom_partitioning

# =========================================================================
# 【核心配置】：为 Shardy 引擎定义的静态 Einsum 切分映射字符串
# 字母含义: b=Batch, t=Time, n=NSize1, m=NSize2, c=Channel
# =========================================================================
BATCH_AXIS_NAME = "data"

# FWD_RULE 含义:
# 输入1(x):        b t n c
# 输入2(h_res_in): b t n m
# 输入3(h_pre_in): b t n
# 输出1(out):      b t c
# 输出2(H_res_out):b t n m
FWD_RULE = "b t n c, b t n m, b t n -> b t c, b t n m"

# BWD_RULE 含义:
# 输入1-3同上
# 输入4(g_out):    b t c
# 输入5(g_H_res):  b t n m
# 输出1(gx):       b t n c
# 输出2(gh_res):   b t n m
# 输出3(gh_pre):   b t n
BWD_RULE = "b t n c, b t n m, b t n, b t c, b t n m -> b t n c, b t n m, b t n"


# 兼容老版本的推导函数 (接收末尾传来的静态参数 *static_args)
def _fwd_infer_sharding(arg_shapes, arg_shardings, *static_args):
    qs = arg_shardings[0]
    return (qs, qs)


def _bwd_infer_sharding(arg_shapes, arg_shardings, *static_args):
    qs = arg_shardings[0]
    return (qs, qs, qs)


# JAX 官方标准的 partition 回调生成器 (接收末尾传来的静态参数 *static_args)
def _create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape, *static_args):
        def lower_fn(*args):
            # 将动态 args 和 静态 static_args 拼在一起传给底层
            return impl_fn(*args, *static_args)

        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition


# =========================================================================


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

    # 准备输出结构 (注意：out 是 BF16/FP16等, H_res_out 是 FP32)
    out_shapes = [
        jax.ShapeDtypeStruct((total_bt, C), x.dtype),  # out_ptr
        jax.ShapeDtypeStruct((total_bt, n, n), jnp.float32),  # H_res_out_ptr
    ]

    # 计算输出步长
    so_bt, so_c = jt.strides_from_shape(out_shapes[0].shape)
    sHr_bt, sHr_n1, sHr_n2 = jt.strides_from_shape(out_shapes[1].shape)

    # 定义 Grid
    def grid(meta):
        return (
            jt.cdiv(total_bt, meta["BLOCK_BT"]),
            jt.cdiv(C, meta["BLOCK_C"]),
        )

    # 调用 Triton
    out_v, H_res_out_v = jt.triton_call(
        x_v,
        h_res_v,
        h_pre_v,  # 指针参数 1, 2, 3
        kernel=sinkhorn_aggregate_fused_kernel,
        out_shape=out_shapes,  # 产生指针参数 4, 5
        grid=grid,
        # --- 标量参数 ---
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


# =========================================================================
# 【新增】：包装前向算子 (适配 SPMD 并行，修复调用方式)
# =========================================================================
def _mhc_pre_op_fwd_spmd_impl(x, h_res_in, h_pre_in, num_iters, eps):
    return mhc_pre_op_fwd_kernel_call(x, h_res_in, h_pre_in, num_iters, eps)


mhc_pre_op_fwd_spmd = custom_partitioning(
    _mhc_pre_op_fwd_spmd_impl, static_argnums=(3, 4)
)

mhc_pre_op_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=_create_partition(mhc_pre_op_fwd_kernel_call),
)


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

    # 调用 Triton
    gx_v, gh_res_v, gh_pre_v = jt.triton_call(
        g_out_v,
        g_H_v,
        x_v,
        h_v,
        h_p_v,  # 输入指针 1, 2, 3, 4, 5
        kernel=sinkhorn_aggregate_bwd_kernel,
        out_shape=out_shapes,  # 产生输出指针 6, 7, 8
        grid=(total_bt, 1),
        # --- 标量参数 ---
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


# =========================================================================
# 【新增】：包装反向算子 (适配 SPMD 并行，修复调用方式)
# =========================================================================
def _mhc_pre_op_bwd_spmd_impl(x, h_res, h_pre, grad_out, grad_H_res, num_iters, eps):
    return mhc_pre_op_bwd_kernel_call(
        x, h_res, h_pre, grad_out, grad_H_res, num_iters, eps
    )


mhc_pre_op_bwd_spmd = custom_partitioning(
    _mhc_pre_op_bwd_spmd_impl, static_argnums=(5, 6)
)

mhc_pre_op_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=_create_partition(mhc_pre_op_bwd_kernel_call),
)


# --- 3. JAX 接口绑定 (解决 Tracer Leak) ---


@partial(jax.jit, static_argnums=(3, 4))
def mhc_pre_op_fused(x, h_res_in, h_pre_in, num_iters=20, eps=1e-8):
    """
    通过闭包捕获静态参数 num_iters 和 eps，确保它们不进入 custom_vjp 的追踪范围。
    """
    C = x.shape[-1]
    if C % 128 != 0:
        raise ValueError(f"mhc_pre_op_fused (triton) requires C % 128 == 0, got C={C}")

    @jax.custom_vjp
    def _internal_op(x_arr, hr_arr, hp_arr):
        # 【修改】：调用切分规则封装后的 FWD
        return mhc_pre_op_fwd_spmd(x_arr, hr_arr, hp_arr, num_iters, eps)

    def _internal_fwd(x_arr, hr_arr, hp_arr):
        out_tuple = mhc_pre_op_fwd_spmd(x_arr, hr_arr, hp_arr, num_iters, eps)
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

        # 【修改】：调用切分规则封装后的 BWD
        gx, ghr, ghp = mhc_pre_op_bwd_spmd(
            x_arr, hr_arr, hp_arr, grad_out, grad_H_res, num_iters, eps
        )
        return gx, ghr, ghp

    _internal_op.defvjp(_internal_fwd, _internal_bwd)

    return _internal_op(x, h_res_in, h_pre_in)
