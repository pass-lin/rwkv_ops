from ..triton_kernel.mhc_post_op import (
    mhc_fused_backward_kernel,
    mhc_fused_forward_kernel,
)
import jax
import jax.numpy as jnp
import jax_triton as jt
import jax.tree_util as jtu

# 引入自定义分区器和切分规则 (适配 JAX 新版 Shardy 引擎)
from jax.experimental.custom_partitioning import custom_partitioning

# =========================================================================
# 【核心配置】：为 Shardy 引擎定义的静态 Einsum 切分映射字符串
# 字母含义: b=Batch, t=Time, c=Channel, n=NSize1, m=NSize2
# =========================================================================
BATCH_AXIS_NAME = "data"

# FWD_RULE 含义:
# 输入1(layer_out):  b t c
# 输入2(x_expanded): b t n c
# 输入3(h_post_raw): b t n
# 输入4(H_res):      b t n m (用独立的 m 避免重复字母报错)
# 输出1(out):        b t n c
FWD_RULE = "b t c, b t n c, b t n, b t n m -> b t n c"

# BWD_RULE 含义:
# 输入1-4同上, 输入5(grad_out): b t n c
# 输出1(gx):       b t c
# 输出2(gh_res):   b t n c
# 输出3(gh_pre):   b t n
# 输出4(grad_H):   b t n m
BWD_RULE = "b t c, b t n c, b t n, b t n m, b t n c -> b t c, b t n c, b t n, b t n m"


# 兼容老版本的推导函数
def _fwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[1]  # 输出与 x_expanded (索引1) shape/sharding 相同
    return (qs,)


def _bwd_infer_sharding(arg_shapes, arg_shardings):
    # 输出梯度对应前 4 个输入
    return (arg_shardings[0], arg_shardings[1], arg_shardings[2], arg_shardings[3])


# JAX 官方标准的 partition 回调生成器
def _create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)

        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition


# =========================================================================


# --- 1. 前向 Launcher ---
def mhc_post_op_fwd_kernel_call(
    layer_out: jax.Array, x_expanded: jax.Array, h_post_raw: jax.Array, H_res: jax.Array
):
    batch, time, NSIZE, channel = x_expanded.shape
    total_bt = batch * time

    # 1. 重塑形状
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
    def grid(meta):
        return (total_bt, jt.cdiv(channel, meta["BLOCK_CHANNEL"]))

    # 4. 调用 Triton
    out_v = jt.triton_call(
        x_v,
        h_v,
        H_v,
        l_v,  # 输入 Array
        kernel=mhc_fused_forward_kernel,
        out_shape=out_struct,
        grid=grid,
        # 标量/步长参数
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


# =========================================================================
# 【新增】：包装前向算子 (适配 SPMD 并行)
# =========================================================================
@custom_partitioning
def mhc_post_op_fwd_spmd(layer_out, x_expanded, h_post_raw, H_res):
    return mhc_post_op_fwd_kernel_call(layer_out, x_expanded, h_post_raw, H_res)


mhc_post_op_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=_create_partition(mhc_post_op_fwd_kernel_call),
)


# --- 2. 反向 Launcher ---
# 【修改】：展平输入参数为 5 个独立 Tensor，方便 SPMD 推导分片规则
def mhc_post_op_bwd_kernel_call(layer_out, x_expanded, h_post_raw, H_res, grad_output):
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

    # 3. 准备输出结构
    out_shapes = [
        jax.ShapeDtypeStruct(x_v.shape, x_v.dtype),  # grad_x
        jax.ShapeDtypeStruct(h_v.shape, jnp.float32),  # grad_h
        jax.ShapeDtypeStruct(H_v.shape, jnp.float32),  # grad_H
        jax.ShapeDtypeStruct(l_v.shape, l_v.dtype),  # grad_l
    ]

    # 4. 计算输出步长
    sgx_bt, sgx_n, sgx_c = sx_bt, sx_n, sx_c
    sgh_bt, sgh_n = sh_bt, sh_n
    sgH_bt, sgH_n1, sgH_n2 = sH_bt, sH_n1, sH_n2
    sgl_bt, sgl_c = sl_bt, sl_c

    # 5. 调用 Triton
    grad_x_v, grad_h_v, grad_H_v, grad_l_v = jt.triton_call(
        x_v,
        h_v,
        H_v,
        l_v,
        g_out_v,
        kernel=mhc_fused_backward_kernel,
        out_shape=out_shapes,
        grid=(total_bt, 1),
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

    # 6. 返回梯度 (注意返回顺序需要和原函数输入参数一致：layer_out, x_expanded, h_post_raw, H_res)
    return (
        grad_l_v.reshape(B, T, C),
        grad_x_v.reshape(B, T, n, C),
        grad_h_v.reshape(B, T, n).astype(h_post_raw.dtype),
        grad_H_v.reshape(B, T, n, n).astype(H_res.dtype),
    )


# =========================================================================
# 【新增】：包装反向算子 (适配 SPMD 并行)
# =========================================================================
@custom_partitioning
def mhc_post_op_bwd_spmd(layer_out, x_expanded, h_post_raw, H_res, grad_output):
    return mhc_post_op_bwd_kernel_call(
        layer_out, x_expanded, h_post_raw, H_res, grad_output
    )


mhc_post_op_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=_create_partition(mhc_post_op_bwd_kernel_call),
)


# --- JAX 接口绑定 ---


@jax.custom_vjp
def mhc_post_op(layer_out, x_expanded, h_post_raw, H_res):
    # 【修改】：调用切分规则封装后的 FWD
    return mhc_post_op_fwd_spmd(
        layer_out.astype(jnp.bfloat16),
        x_expanded.astype(jnp.bfloat16),
        h_post_raw.astype(jnp.float32),
        H_res.astype(jnp.float32),
    )


def mhc_post_op_fwd(layer_out, x_expanded, h_post_raw, H_res):
    layer_out_c = layer_out.astype(jnp.bfloat16)
    x_expanded_c = x_expanded.astype(jnp.bfloat16)
    h_post_raw_c = h_post_raw.astype(jnp.float32)
    H_res_c = H_res.astype(jnp.float32)
    out = mhc_post_op_fwd_spmd(layer_out_c, x_expanded_c, h_post_raw_c, H_res_c)
    # 保存 cast 后的张量，确保反向使用的 dtype 与前向实际进 kernel 的 dtype 一致
    return out, (layer_out_c, x_expanded_c, h_post_raw_c, H_res_c)


def mhc_post_op_bwd(res, grad_output):
    layer_out, x_expanded, h_post_raw, H_res = res

    # 保证健壮性，若上游传了 None 梯度则初始化为零张量
    if grad_output is None:
        grad_output = jnp.zeros_like(x_expanded)

    # 【修改】：调用切分规则封装后的 BWD，展平传递参数
    grads = mhc_post_op_bwd_spmd(layer_out, x_expanded, h_post_raw, H_res, grad_output)
    return grads


mhc_post_op.defvjp(mhc_post_op_fwd, mhc_post_op_bwd)
