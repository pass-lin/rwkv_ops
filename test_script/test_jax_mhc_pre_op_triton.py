import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

os.environ["KERNEL_TYPE"] = "triton"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KERAS_BACKEND"] = "jax"
# 1. 导入你的 JAX-Triton 算子
# 假设已经按照之前的指南定义并导出
from rwkv_ops.mhc_kernel.jax_triton_op.mhc_pre_op import (
    mhc_pre_op_fused as triton_mhc_pre_op,
)
from rwkv_ops.mhc_kernel.native_op import mhc_pre_op_fused as native_mhc_pre_op

B, T, n, C = 256, 512, 4, 512
dtype = jnp.bfloat16

# JAX 随机数管理
key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)


def get_inputs():
    x = jax.random.normal(k1, (B, T, n, C)).astype(jnp.bfloat16)
    h_res = jax.random.normal(k2, (B, T, n, n)).astype(jnp.float32)
    h_pre = jax.random.normal(k3, (B, T, n)).astype(jnp.float32)
    return x, h_res, h_pre


def test_is_close(name, x1, x2, atol=1e-2, rtol=1e-2):
    if x1 is None or x2 is None:
        print(f"❌ {name} 数据不存在")
        return

    x1_np = np.array(x1, dtype=np.float32)
    x2_np = np.array(x2, dtype=np.float32)

    error = np.abs(x1_np - x2_np)
    avg_error = error.mean()
    rel_error = avg_error / (np.abs(x1_np).mean() + 1e-9)

    print(f"[{name}]")
    print(f"  平均绝对误差: {avg_error:.6e}")
    print(f"  平均相对误差: {rel_error:.6e}")

    # 计算完全一样的比例 (在 BF16 下通常较低，主要看 max_diff)
    equal_rate = np.sum(error < 1e-6) / x1_np.size * 100
    print(f"  {equal_rate:.2f}% 的数据在 1e-6 误差内一致")

    if np.isnan(x1_np).any() or np.isnan(x2_np).any():
        print(f"  ❌❌ 存在 NaN ❌❌")
        return

    try:
        np.testing.assert_allclose(x1_np, x2_np, atol=atol, rtol=rtol)
        print(f"  ✅ 数值一致 (max_diff: {error.max():.6e})")
    except AssertionError as e:
        print(f"  ❌ 数值不一致")


print("--- 开始数值正确性校验 ---")

x_data, hr_data, hp_data = get_inputs()


# 定义 Loss 函数用于测试反向梯度
def loss_fn(op_func, x, hr, hp):
    out_x, out_h = op_func(x, hr, hp, num_iters=20)
    return jnp.mean(out_x.astype(jnp.float32) ** 2) + jnp.mean(out_h**2)


# --- 前向对比 ---
out_x_n, out_h_n = native_mhc_pre_op(x_data, hr_data, hp_data)
out_x_t, out_h_t = triton_mhc_pre_op(x_data, hr_data, hp_data)

test_is_close("Forward: x_layer_in", out_x_n, out_x_t)
test_is_close("Forward: H_res", out_h_n, out_h_t)

# --- 反向梯度对比 ---
grad_native_fn = jax.grad(partial(loss_fn, native_mhc_pre_op), argnums=(0, 1, 2))
grad_triton_fn = jax.grad(partial(loss_fn, triton_mhc_pre_op), argnums=(0, 1, 2))

print("\n--- 开始梯度对比 ---")
gx_n, ghr_n, ghp_n = grad_native_fn(x_data, hr_data, hp_data)
gx_t, ghr_t, ghp_t = grad_triton_fn(x_data, hr_data, hp_data)

test_is_close("Gradient: x", gx_n, gx_t)
test_is_close("Gradient: h_res_in", ghr_n, ghr_t)
test_is_close("Gradient: h_pre_in", ghp_n, ghp_t)

print("\n" + "=" * 40)
print("🚀 开始全流程性能基准测试 (Forward + Backward)")
print("=" * 40)

n_warmup = 10
n_repeat = 100

# JAX 需要显式 jit
jit_native_full = jax.jit(
    jax.value_and_grad(partial(loss_fn, native_mhc_pre_op), argnums=(0, 1, 2))
)
jit_triton_full = jax.jit(
    jax.value_and_grad(partial(loss_fn, triton_mhc_pre_op), argnums=(0, 1, 2))
)


def benchmark_all(run_fn, name, x, hr, hp):
    # 预热
    for _ in range(n_warmup):
        res = run_fn(x, hr, hp)
        jax.block_until_ready(res)

    start_time = time.perf_counter()
    for _ in range(n_repeat):
        res = run_fn(x, hr, hp)
        jax.block_until_ready(res)
    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) / n_repeat * 1000
    print(f"👉 {name:20s} Full (Fwd+Bwd): {avg_time_ms:.4f} ms")
    return avg_time_ms


# 运行 Benchmark
print(f"配置: B={B}, T={T}, n={n}, C={C}, dtype={dtype}")

# 1. Native Eager (JAX 极慢，不建议跑太大型的 B, T)
# t_native_eager = benchmark_all(lambda x, hr, hp: jax.grad(partial(loss_fn, native_mhc_pre_op))(x, hr, hp), "Native (Eager)", x_data, hr_data, hp_data)

t_native_jit = benchmark_all(
    jax.jit(native_mhc_pre_op), "JAX JIT (Native)", x_data, hr_data, hp_data
)

# 3. Triton Fused JIT
t_triton_jit = benchmark_all(
    jax.jit(triton_mhc_pre_op), "JAX-Triton Fused", x_data, hr_data, hp_data
)

print("\n📊 性能提升总结:")
print(f"Triton vs JIT Native前向: {t_native_jit / t_triton_jit:.2f}x faster")
t_native_jit = benchmark_all(
    jit_native_full, "JAX JIT (Native)", x_data, hr_data, hp_data
)

# 3. Triton Fused JIT
t_triton_jit = benchmark_all(
    jit_triton_full, "JAX-Triton Fused", x_data, hr_data, hp_data
)


print(f"Triton vs JIT Native: {t_native_jit / t_triton_jit:.2f}x faster")
# if 't_native_eager' in locals():
#     print(f"Triton vs Eager:      {t_native_eager / t_triton_jit:.2f}x faster")
print("\n🎉🎉 MHC Pre-Op 全校验结束 🎉🎉")
