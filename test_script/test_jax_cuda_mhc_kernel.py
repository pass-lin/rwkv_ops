import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 请根据实际情况修改
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
    """精度对比辅助函数"""
    x1_val = np.array(x1)
    x2_val = np.array(x2)
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
        print(f"❌ {name:25} Fail (Max Diff: {max_diff:.6e})")
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
raise (1)
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
    partial(mix_loss, jax_mhc), has_aux=True, argnums=(1, 2)
)(x_mix, m_mat)
(l2, out_nat), g_nat = jax.value_and_grad(
    partial(mix_loss, native_mhc), has_aux=True, argnums=(1, 2)
)(x_mix, m_mat)
check_close("StreamMix Forward", out_jax, out_nat)
check_close("StreamMix Grad: dx", g_jax[0], g_nat[0])
check_close("StreamMix Grad: dM", g_jax[1], g_nat[1])

# =====================================================
# 4. Stream Aggregate 测试
# =====================================================
print(f"\n{' Stream Aggregate 测试 ':=^50}")
h_pre = rand_f32(key, (B, T, n))


def agg_loss(m, x, h):
    out = m.stream_aggregate(x, h)
    return jnp.sum(out.astype(jnp.float32)), out


(l1, out_jax), g_jax = jax.value_and_grad(
    partial(agg_loss, jax_mhc), has_aux=True, argnums=(1, 2)
)(x_mix, h_pre)
(l2, out_nat), g_nat = jax.value_and_grad(
    partial(agg_loss, native_mhc), has_aux=True, argnums=(1, 2)
)(x_mix, h_pre)
check_close("StreamAgg Forward", out_jax, out_nat)
check_close("StreamAgg Grad: dx", g_jax[0], g_nat[0])

# =====================================================
# 5. Stream Distribute 测试
# =====================================================
print(f"\n{' Stream Distribute 测试 ':=^50}")
l_out = rand_bfp(key, (B, T, C))
h_post = rand_f32(key, (B, T, n))


def dist_loss(m, l, h):
    out = m.stream_distribute(l, h)
    return jnp.sum(out.astype(jnp.float32)), out


(l1, out_jax), g_jax = jax.value_and_grad(
    partial(dist_loss, jax_mhc), has_aux=True, argnums=(1, 2)
)(l_out, h_post)
(l2, out_nat), g_nat = jax.value_and_grad(
    partial(dist_loss, native_mhc), has_aux=True, argnums=(1, 2)
)(l_out, h_post)
check_close("StreamDist Forward", out_jax, out_nat)
check_close("StreamDist Grad: dl", g_jax[0], g_nat[0])

# =====================================================
# 6. MHC Pre-Op (Fused) 测试
# =====================================================
print(f"\n{' mHC Pre-Op (Fused) 测试 ':=^50}")
h_res_raw = rand_f32(key, (B, T, n * n))


def pre_op_loss(m, x, h1, h2, hr):
    # 返回 (x_layer_in, H_post, H_res)
    x_in, hp, h_res = m.mhc_pre_op(x, h1, h2, hr, num_iters=20, eps=1e-8)
    return jnp.sum(x_in.astype(jnp.float32)) + jnp.sum(hp) + jnp.sum(h_res), (
        x_in,
        hp,
        h_res,
    )


(l1, out_jax), g_jax = jax.value_and_grad(
    partial(pre_op_loss, jax_mhc), has_aux=True, argnums=(1, 2, 3, 4)
)(x_mix, h_pre, h_post, h_res_raw)
(l2, out_nat), g_nat = jax.value_and_grad(
    partial(pre_op_loss, native_mhc), has_aux=True, argnums=(1, 2, 3, 4)
)(x_mix, h_pre, h_post, h_res_raw)
check_close("PreOp Fwd: x_layer_in", out_jax[0], out_nat[0])
check_close("PreOp Grad: dx_exp", g_jax[0], g_nat[0])

# =====================================================
# 7. MHC Post-Op (Fused) 测试
# =====================================================
print(f"\n{' mHC Post-Op (Fused) 测试 ':=^50}")
h_res_mat = rand_f32(key, (B, T, n, n))


def post_op_loss(m, l, x, hp, hr):
    out = m.mhc_post_op(l, x, hp, hr)
    return jnp.sum(out.astype(jnp.float32)), out


(l1, out_jax), g_jax = jax.value_and_grad(
    partial(post_op_loss, jax_mhc), has_aux=True, argnums=(1, 2, 3, 4)
)(l_out, x_mix, h_post, h_res_mat)
(l2, out_nat), g_nat = jax.value_and_grad(
    partial(post_op_loss, native_mhc), has_aux=True, argnums=(1, 2, 3, 4)
)(l_out, x_mix, h_post, h_res_mat)
check_close("PostOp Forward", out_jax, out_nat)
check_close("PostOp Grad: dl_out", g_jax[0], g_nat[0])

print(f"\n{' JAX MHC 7个算子全部验证完成 ':=^50}")
