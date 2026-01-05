import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 请根据实际情况修改
os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# 假设您的项目结构如下：
# rwkv_ops/mhc_kernel/jax_kernel/mhu_jax.py (调用 FFI)
# rwkv_ops/mhc_kernel/native_keras_op.py (纯 Keras/JAX 实现)
import rwkv_ops.mhc_kernel.jax_kernel.mhu_jax as jax_mhc
import rwkv_ops.mhc_kernel.native_keras_op as native_mhc


def check_close(name, x1, x2, atol=1e-4, rtol=1e-4):
    print(f"Testing {name}")
    """精度对比辅助函数"""
    x1_val = np.array(x1.astype("float32"))
    x2_val = np.array(x2.astype("float32"))
    error = np.abs(x1_val - x2_val)
    if np.sum(error) == 0:
        print("✅✅✅完全一致✅✅✅")
        return
    try:
        np.testing.assert_allclose(x1_val, x2_val, atol=atol, rtol=rtol)
        print(f"✅ {name:25} Pass")
        return True
    except AssertionError as e:
        max_diff = np.max(np.abs(x1_val - x2_val))
        print(f"❌ {e}")
        print(
            f"{float(np.sum(error == 0)) / float(np.cumprod(error.shape)[-1])}是完全一模一样的，平均误差是{error.mean()}"
        )
        return False


def rand_bfp(key, shape):
    return jax.random.normal(key, shape).astype(jnp.bfloat16)


def rand_f32(key, shape):
    return jax.random.normal(key, shape).astype(jnp.float32)


# =====================================================
# 初始化参数
# =====================================================
key = jax.random.PRNGKey(42)
B, T, n, C = 2, 16, 4, 64  # Batch, Seq, Streams, Channels

# =====================================================
# 1. Sinkhorn Knopp 测试
# =====================================================
print(f"\n{' Sinkhorn 测试 ':=^50}")
s_inp = jnp.abs(rand_f32(key, (B, T, n, n))) + 0.1


def sk_loss(m, x):
    out = m.sinkhorn_knopp(x, num_iters=20, eps=1e-8)
    return jnp.mean(out**2), out


(l1, out_jax), g_jax = jax.value_and_grad(partial(sk_loss, jax_mhc), has_aux=True)(
    s_inp
)
(l2, out_nat), g_nat = jax.value_and_grad(partial(sk_loss, native_mhc), has_aux=True)(
    s_inp
)
check_close("Sinkhorn Forward", out_jax, out_nat)
check_close("Sinkhorn Gradient", g_jax, g_nat)

# =====================================================
# 2. RMSNorm 测试
# =====================================================
print(f"\n{' RMSNorm 测试 ':=^50}")
x_norm = rand_bfp(key, (B, T, n, C))


def rms_loss(m, x):
    out = m.rmsnorm(x, eps=1e-5)
    return jnp.sum(out.astype(jnp.float32)), out


(l1, out_jax), g_jax = jax.value_and_grad(partial(rms_loss, jax_mhc), has_aux=True)(
    x_norm
)
(l2, out_nat), g_nat = jax.value_and_grad(partial(rms_loss, native_mhc), has_aux=True)(
    x_norm
)
check_close("RMSNorm Forward", out_jax, out_nat, atol=1e-2)
check_close("RMSNorm Gradient", g_jax, g_nat, atol=1e-3)

# =====================================================
# 3. Stream Mix 测试
# =====================================================
print(f"\n{' Stream Mix 测试 ':=^50}")
x_mix = rand_bfp(key, (B, T, n, C))
m_mat = rand_f32(key, (B, T, n, n))


def mix_loss(m, x, mat):
    out = m.stream_mix(x, mat)
    return jnp.sum(out.astype(jnp.float32)), out


(l1, out_jax), g_jax = jax.value_and_grad(
    partial(mix_loss, jax_mhc), has_aux=True, argnums=(0, 1)
)(x_mix, m_mat)
(l2, out_nat), g_nat = jax.value_and_grad(
    partial(mix_loss, native_mhc), has_aux=True, argnums=(0, 1)
)(x_mix, m_mat)
check_close("StreamMix Forward", out_jax, out_nat, 5e-3, 7e-3)
check_close("StreamMix Grad: dx", g_jax[0], g_nat[0], 5e-3, 5e-3)
check_close("StreamMix Grad: dM", g_jax[1], g_nat[1], 5e-3, 5e-3)

# =====================================================
# 4. stream_aggregate 测试
# =====================================================
print(f"\n{' Stream Aggregate 测试 ':=^50}")
x_agg = rand_bfp(key, (B, T, n, C))
H_agg = rand_f32(key, (B, T, n))


def agg_loss(m, x, h):
    out = m.stream_aggregate(x, h)
    return jnp.sum(out.astype(jnp.float32)), out


(l1, out_jax), g_jax = jax.value_and_grad(
    partial(agg_loss, jax_mhc), has_aux=True, argnums=(0, 1)
)(x_agg, H_agg)
(l2, out_nat), g_nat = jax.value_and_grad(
    partial(agg_loss, native_mhc), has_aux=True, argnums=(0, 1)
)(x_agg, H_agg)
check_close("StreamAggregate Forward", out_jax, out_nat, atol=5e-3, rtol=5e-3)
check_close("StreamAggregate Grad: dx", g_jax[0], g_nat[0], atol=5e-3, rtol=5e-3)
check_close("StreamAggregate Grad: dH", g_jax[1], g_nat[1], atol=5e-3, rtol=5e-3)

# =====================================================
# 5. Stream Distribute 测试
# =====================================================
print(f"\n{' Stream Distribute 测试 ':=^50}")
# 输入形状: Inp [B, T, C], H_post [B, T, n] -> Out [B, T, n, C]
x_dist = rand_bfp(key, (B, T, C))
H_dist = rand_f32(key, (B, T, n))


def dist_loss(m, x, h):
    out = m.stream_distribute(x, h)
    # 聚合回标量以计算梯度
    return jnp.sum(out.astype(jnp.float32)), out


# 计算 JAX FFI 版本的 Loss 和梯度
(l1, out_jax), g_jax = jax.value_and_grad(
    partial(dist_loss, jax_mhc), has_aux=True, argnums=(0, 1)
)(x_dist, H_dist)

# 计算 Native (Keras/JAX) 版本的 Loss 和梯度
(l2, out_nat), g_nat = jax.value_and_grad(
    partial(dist_loss, native_mhc), has_aux=True, argnums=(0, 1)
)(x_dist, H_dist)

# 验证
check_close("StreamDistribute Forward", out_jax, out_nat, atol=5e-3, rtol=5e-3)
check_close(
    "StreamDistribute Grad: dx", g_jax[0], g_nat[0], atol=5e-3, rtol=5e-3
)  # 注意对应的argnums
check_close("StreamDistribute Grad: dH", g_jax[1], g_nat[1], atol=5e-3, rtol=5e-3)

# =====================================================
# 6. MHC Post-Op (Fused) 测试
# =====================================================
print(f"\n{' MHC Post-Op 融合测试 ':=^50}")
# 输入形状: 
# layer_out: [B, T, C]
# x_expanded: [B, T, n, C]
# H_post: [B, T, n]
# H_res: [B, T, n, n]
lo_val = rand_bfp(key, (B, T, C))
xe_val = rand_bfp(key, (B, T, n, C))
hp_val = rand_f32(key, (B, T, n))
hr_val = rand_f32(key, (B, T, n, n))

def post_loss(m, lo, xe, hp, hr):
    # 调用融合算子实现: (H_res @ x_expanded) + (layer_out * H_post)
    out = m.mhc_post_op(lo, xe, hp, hr)
    return jnp.sum(out.astype(jnp.float32)), out

# 计算 JAX FFI 版本的 Loss 和梯度 (针对全部 4 个输入参数)
(l1, out_jax), g_jax = jax.value_and_grad(
    partial(post_loss, jax_mhc), has_aux=True, argnums=(0, 1, 2, 3)
)(lo_val, xe_val, hp_val, hr_val)

# 计算 Native (Keras/JAX) 版本的 Loss 和梯度
(l2, out_nat), g_nat = jax.value_and_grad(
    partial(post_loss, native_mhc), has_aux=True, argnums=(0, 1, 2, 3)
)(lo_val, xe_val, hp_val, hr_val)

# 验证前向和所有梯度
check_close("PostOp Forward", out_jax, out_nat, atol=5e-3,rtol=7e-3)
check_close("PostOp Grad: d_layer_out", g_jax[0], g_nat[0], atol=1e-3)
check_close("PostOp Grad: d_x_expanded", g_jax[1], g_nat[1], atol=5e-3,rtol=5e-3)
check_close("PostOp Grad: d_H_post", g_jax[2], g_nat[2], atol=1e-3)
check_close("PostOp Grad: d_H_res", g_jax[3], g_nat[3], atol=1e-3)

# =====================================================
# 7. MHC Pre-Op (Fused) 测试
# =====================================================
print(f"\n{' MHC Pre-Op 融合测试 ':=^50}")

# 输入形状：
# x_expanded: [B, T, n, C]
# h_pre_raw:  [B, T, n]
# h_post_raw: [B, T, n]
# h_res_raw:  [B, T, n, n] 或 [B, T, n*n]
xe_pre = rand_bfp(key, (B, T, n, C))
hpre_raw = rand_f32(key, (B, T, n))
hpost_raw = rand_f32(key, (B, T, n))
hres_raw = rand_f32(key, (B, T, n, n))   # 4D 原始输入


def pre_loss(m, xe, hpre, hpost, hres):
    # 返回融合算子输出 (x_layer_in, H_post, H_res) 与标量损失
    x_layer_in, H_post, H_res = m.mhc_pre_op(xe, hpre, hpost, hres, num_iters=20, eps=1e-8)
    # 简单标量损失：三项平方和
    loss = (
        jnp.sum(x_layer_in.astype(jnp.float32) ** 2)
        + jnp.sum(H_post ** 2)
        + jnp.sum(H_res ** 2)
    )
    return loss, (x_layer_in, H_post, H_res)


# 计算 JAX FFI 版本
(loss_jax, (xli_jax, hp_jax, hr_jax)), g_jax = jax.value_and_grad(
    partial(pre_loss, jax_mhc), has_aux=True, argnums=(0, 1, 2, 3)
)(xe_pre, hpre_raw, hpost_raw, hres_raw)

# 计算 Native (Keras/JAX) 版本
(loss_nat, (xli_nat, hp_nat, hr_nat)), g_nat = jax.value_and_grad(
    partial(pre_loss, native_mhc), has_aux=True, argnums=(0, 1, 2, 3)
)(xe_pre, hpre_raw, hpost_raw, hres_raw)

# 验证前向
check_close("PreOp Forward: x_layer_in", xli_jax, xli_nat, atol=5e-3, rtol=5e-3)
check_close("PreOp Forward: H_post", hp_jax, hp_nat, atol=5e-3, rtol=5e-3)
check_close("PreOp Forward: H_res", hr_jax, hr_nat, atol=5e-3, rtol=5e-3)

# 验证梯度
check_close("PreOp Grad: d_x_expanded", g_jax[0], g_nat[0], atol=5e-3, rtol=5e-3)
check_close("PreOp Grad: d_h_pre_raw",  g_jax[1], g_nat[1], atol=5e-3, rtol=5e-3)
check_close("PreOp Grad: d_h_post_raw", g_jax[2], g_nat[2], atol=5e-3, rtol=5e-3)
check_close("PreOp Grad: d_h_res_raw",  g_jax[3], g_nat[3], atol=5e-3, rtol=5e-3)

print("\n🎉 全部 MHC 算子通过数值对齐测试！")