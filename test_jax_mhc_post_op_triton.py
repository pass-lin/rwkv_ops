import os

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERNEL_TYPE"] = "triton"
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from rwkv_ops.mhc_kernel.jax_triton_op.mhc_post_op import mhc_post_op as triton_mhc_op
from rwkv_ops.mhc_kernel.native_op import mhc_post_op as native_mhc_op

# ------------------------------------------------------------------
# 1. 配置参数与构造输入
# ------------------------------------------------------------------
B, T, n, C = 256, 256, 4, 512
dtype = jnp.bfloat16

# JAX 使用 PRNGKey 管理随机性
key = jax.random.PRNGKey(42)
k1, k2, k3, k4 = jax.random.split(key, 4)

# 创建随机输入
layer_out_raw = jax.random.normal(k1, (B, T, C)).astype(jnp.bfloat16)
x_expanded_raw = jax.random.normal(k2, (B, T, n, C)).astype(jnp.bfloat16)
h_post_raw_raw = jax.random.normal(k3, (B, T, n)).astype(jnp.float32)
H_res_raw = jax.random.normal(k4, (B, T, n, n)).astype(jnp.float32)


# ------------------------------------------------------------------
# 2. 工具函数：数值比较
# ------------------------------------------------------------------
def test_is_close(name, x1, x2, atol=1e-2, rtol=1e-2):
    x1_np = np.array(x1, dtype=np.float32)
    x2_np = np.array(x2, dtype=np.float32)
    error = np.abs(x1_np - x2_np)
    print(
        f"平均绝对误差是{error.mean()},平均相对误差是{error.mean() / np.abs(x1_np).mean()}"
    )
    print(f"{np.sum(error < 1e-8) / np.cumprod(x1_np.shape)[-1] * 100}%的数据完全一样")
    if np.isnan(x1_np).any() or np.isnan(x2_np).any():
        print(f"❌❌ {name} 存在 NaN ❌❌")
        return

    try:
        np.testing.assert_allclose(x1_np, x2_np, atol=atol, rtol=rtol)
        print(f"✅ {name} 数值一致 (max_diff: {np.abs(x1_np - x2_np).max():.6f})")
    except AssertionError as e:
        print(f"❌ {name} 数值不一致")
        print(e)


# ------------------------------------------------------------------
# 3. 精度校验 (Forward & Backward)
# ------------------------------------------------------------------
print("--- 开始前向测试 ---")
output_triton = triton_mhc_op(layer_out_raw, x_expanded_raw, h_post_raw_raw, H_res_raw)
output_native = native_mhc_op(layer_out_raw, x_expanded_raw, h_post_raw_raw, H_res_raw)

test_is_close("Forward Output", output_native, output_triton)

print("\n--- 开始反向测试 ---")


# 定义一个计算 Loss 的函数用于求导
def loss_fn(op_func, l, x, h, H):
    out = op_func(l, x, h, H)
    return jnp.mean(out.astype(jnp.float32) ** 2)


# 使用 jax.value_and_grad 获取 loss 和所有输入的梯度
grad_native_fn = jax.value_and_grad(
    partial(loss_fn, native_mhc_op), argnums=(0, 1, 2, 3)
)
grad_triton_fn = jax.value_and_grad(
    partial(loss_fn, triton_mhc_op), argnums=(0, 1, 2, 3)
)

loss_native, grads_native = grad_native_fn(
    layer_out_raw, x_expanded_raw, h_post_raw_raw, H_res_raw
)
loss_triton, grads_triton = grad_triton_fn(
    layer_out_raw, x_expanded_raw, h_post_raw_raw, H_res_raw
)

grad_names = ["layer_out", "x_expanded", "h_post_raw", "H_res"]
for i, name in enumerate(grad_names):
    test_is_close(f"Gradient: {name}", grads_native[i], grads_triton[i])

# ------------------------------------------------------------------
# 4. 性能基准测试 (Benchmark)
# ------------------------------------------------------------------
print("\n" + "=" * 40)
print("🚀 开始性能基准测试 (Speed Benchmark)")
print("=" * 40)

n_warmup = 10
n_repeat = 100

# 准备编译版本
# 注意：triton_mhc_op 内部已经包含了 triton_call，
# 我们将其放入 jax.jit 以确保整个 Dispatch 过程被优化
jit_native_op = jax.jit(native_mhc_op)
jit_triton_op = jax.jit(triton_mhc_op)


def benchmark_op(op_func, name, l, x, h, H, is_backward=False):
    # 封装函数以便于统一求导或直接执行
    if is_backward:
        # 这里的函数包含前向和反向
        run_fn = jax.jit(jax.grad(partial(loss_fn, op_func), argnums=(0, 1, 2, 3)))
    else:
        run_fn = jax.jit(op_func) if "jit" in name.lower() else op_func

    # 预热 (Warmup)
    for _ in range(n_warmup):
        res = run_fn(l, x, h, H)
        # JAX 是异步执行的，必须 block 确保计算完成
        jax.block_until_ready(res)

    # 计时
    start_time = time.perf_counter()
    for _ in range(n_repeat):
        res = run_fn(l, x, h, H)
        jax.block_until_ready(res)
    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) / n_repeat * 1000
    label = "Backward (+Fwd)" if is_backward else "Forward"
    print(f"👉 {name:20s} {label}: {avg_time_ms:.4f} ms")
    return avg_time_ms


# 运行测试
print(f"配置: B={B}, T={T}, n={n}, C={C}, dtype={dtype}")

print("\n--- Forward Speed ---")
# JAX Eager 模式通常非常慢，仅供参考
t_eager_fwd = benchmark_op(
    native_mhc_op, "JAX Eager", layer_out_raw, x_expanded_raw, h_post_raw_raw, H_res_raw
)
t_jit_fwd = benchmark_op(
    jit_native_op,
    "JAX JIT (Native)",
    layer_out_raw,
    x_expanded_raw,
    h_post_raw_raw,
    H_res_raw,
)
t_triton_fwd = benchmark_op(
    jit_triton_op,
    "JAX-Triton JIT",
    layer_out_raw,
    x_expanded_raw,
    h_post_raw_raw,
    H_res_raw,
)

print("\n--- Backward Speed (Forward + Backward) ---")
# 注意：JAX 的反向传播必须在编译函数上进行才有意义
t_jit_bwd = benchmark_op(
    native_mhc_op,
    "JAX JIT (Native)",
    layer_out_raw,
    x_expanded_raw,
    h_post_raw_raw,
    H_res_raw,
    is_backward=True,
)
t_triton_bwd = benchmark_op(
    triton_mhc_op,
    "JAX-Triton JIT",
    layer_out_raw,
    x_expanded_raw,
    h_post_raw_raw,
    H_res_raw,
    is_backward=True,
)

# ------------------------------------------------------------------
# 5. 总结分析
# ------------------------------------------------------------------
print("\n" + "=" * 40)
print("📊 性能提升总结 (Speedup)")
print("=" * 40)


def print_speedup(base_name, base_time, target_name, target_time):
    speedup = base_time / target_time
    print(f"{target_name} vs {base_name}: {speedup:.2f}x faster")


print(">>> Forward:")
print_speedup("JAX JIT Native", t_jit_fwd, "JAX-Triton", t_triton_fwd)

print("\n>>> Backward:")
print_speedup("JAX JIT Native", t_jit_bwd, "JAX-Triton", t_triton_bwd)

print("\n🎉🎉 MHC JAX-Triton 算子校验结束 🎉🎉")
