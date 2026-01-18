import os
import torch
import numpy as np

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 1. 导入算子
from rwkv_ops.mhc_kernel.native_op import (
    sinkhorn_knopp as native_sinkhorn,
    stream_aggregate as native_aggregate,
)
from rwkv_ops.mhc_kernel.torch_triton_op.mhc_pre_op import mhc_pre_op_fused

# ------------------------------------------------------------------
# 1. 配置参数与构造输入
# ------------------------------------------------------------------
B, T, n, C = 256, 256, 4, 768
device = "cuda"
dtype = torch.bfloat16


# 构造输入函数 (用于刷新梯度空间)
def get_inputs():
    x = torch.randn(B, T, n, C, device=device, dtype=dtype).requires_grad_(True)
    h_res = torch.randn(B, T, n, n, device=device, dtype=torch.float32).requires_grad_(
        True
    )
    h_pre = torch.randn(B, T, n, device=device, dtype=torch.float32).requires_grad_(
        True
    )
    return x, h_res, h_pre


# ------------------------------------------------------------------
# 2. 工具函数：数值比较
# ------------------------------------------------------------------
def test_is_close(name, x1, x2, atol=1e-2, rtol=1e-2):
    if x1 is None or x2 is None:
        print(f"❌ {name} 梯度不存在")
        return

    x1_np = x1.detach().cpu().float().numpy()
    x2_np = x2.detach().cpu().float().numpy()

    error = np.abs(x1_np - x2_np)
    avg_error = error.mean()
    rel_error = avg_error / (np.abs(x1_np).mean() + 1e-9)

    print(f"[{name}]")
    print(f"  平均绝对误差: {avg_error:.6e}")
    print(f"  平均相对误差: {rel_error:.6e}")

    equal_rate = np.sum(error < 1e-7) / x1_np.size * 100
    print(f"  {equal_rate:.2f}% 的数据完全一样")

    if np.isnan(x1_np).any() or np.isnan(x2_np).any():
        print(f"  ❌❌ 存在 NaN ❌❌")
        return

    try:
        np.testing.assert_allclose(x1_np, x2_np, atol=atol, rtol=rtol)
        print(f"  ✅ 数值一致 (max_diff: {error.max():.6e})")
    except AssertionError as e:
        print(f"  ❌ 数值不一致")
        # print(e)


# ------------------------------------------------------------------
# 3. 正确性校验 (Forward & Backward)
# ------------------------------------------------------------------
print("--- 开始数值正确性校验 ---")

# --- Native Pass ---
x_n, hr_n, hp_n = get_inputs()
h_res_n_out = native_sinkhorn(hr_n, num_iters=20)
x_in_n_out = native_aggregate(x_n, hp_n)

# 构造 Loss 以测试反向
loss_n = (x_in_n_out.float() ** 2).mean() + (h_res_n_out**2).mean()
loss_n.backward()

# --- Triton Pass ---
x_t, hr_t, hp_t = get_inputs()
# 拷贝同样的数值以确保起点一致
with torch.no_grad():
    x_t.copy_(x_n)
    hr_t.copy_(hr_n)
    hp_t.copy_(hp_n)

x_in_t_out, h_res_t_out = mhc_pre_op_fused(x_t, hr_t, hp_t, num_iters=20)

loss_t = (x_in_t_out.float() ** 2).mean() + (h_res_t_out**2).mean()
loss_t.backward()

# 比较前向
test_is_close("Forward: x_layer_in", x_in_n_out, x_in_t_out)
test_is_close("Forward: H_res", h_res_n_out, h_res_t_out)

# 比较反向梯度
test_is_close("Gradient: x", x_n.grad, x_t.grad)
test_is_close("Gradient: h_res_in", hr_n.grad, hr_t.grad)
test_is_close("Gradient: h_pre_in", hp_n.grad, hp_t.grad)

# ------------------------------------------------------------------
# 4. 性能基准测试 (Benchmark)
# ------------------------------------------------------------------
print("\n" + "=" * 40)
print("🚀 开始全流程性能基准测试 (Forward + Backward)")
print("=" * 40)

n_warmup = 10
n_repeat = 100


# Native 逻辑包装
def native_full_op(x, hr, hp):
    h_out = native_sinkhorn(hr, num_iters=20)
    x_out = native_aggregate(x, hp)
    return x_out, h_out


# torch.compile 逻辑包装
@torch.compile
def compiled_full_op(x, hr, hp):
    h_out = native_sinkhorn(hr, num_iters=20)
    x_out = native_aggregate(x, hp)
    return x_out, h_out


def benchmark_all(op_func, name, x, hr, hp):
    # 预热
    for _ in range(n_warmup):
        out_x, out_h = op_func(x, hr, hp)
        loss = (out_x.float() ** 2).mean() + (out_h**2).mean()
        loss.backward()
        x.grad, hr.grad, hp.grad = None, None, None
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_repeat):
        out_x, out_h = op_func(x, hr, hp)
        loss = (out_x.float() ** 2).mean() + (out_h**2).mean()
        loss.backward()
        x.grad, hr.grad, hp.grad = None, None, None
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) / n_repeat
    print(f"👉 {name:20s} Full (Fwd+Bwd): {elapsed_time_ms:.4f} ms")
    return elapsed_time_ms


# 运行 Benchmark
x, hr, hp = get_inputs()
print(f"配置: B={B}, T={T}, n={n}, C={C}, dtype={dtype}")

t_native = benchmark_all(native_full_op, "Native (Python)", x, hr, hp)
t_compile = benchmark_all(compiled_full_op, "torch.compile", x, hr, hp)
t_triton = benchmark_all(mhc_pre_op_fused, "Triton Fused", x, hr, hp)

print("\n📊 性能提升总结:")
print(f"Triton vs Native:  {t_native / t_triton:.2f}x faster")
print(f"Triton vs Compile: {t_compile / t_triton:.2f}x faster")

print("\n🎉🎉 MHC Pre-Op 全校验结束 🎉🎉")
