import os
import torch
import numpy as np
from keras import ops

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 导入算子
from rwkv_ops.mhc_kernel.native_op import mhc_post_op as native_mhc_op
from rwkv_ops.mhc_kernel.torch_triton_op.mhc_post_op import mhc_post_op as triton_mhc_op

# ------------------------------------------------------------------
# 1. 配置参数与构造输入
# ------------------------------------------------------------------
B, T, n, C = 4, 64, 4, 512
device = "cuda"
dtype = torch.bfloat16

# 创建随机输入
# 为保证公平对比，我们先创建全精度的数据，再转为目标类型
layer_out_raw = torch.randn(B, T, C, device=device).bfloat16()
x_expanded_raw = torch.randn(B, T, n, C, device=device).bfloat16()
h_post_raw_raw = torch.randn(B, T, n, device=device).float()
H_res_raw = torch.randn(B, T, n, n, device=device).float()

# Triton 版本的输入 (BF16/FP32 混合)
# 注意：按照我们的封装，x 和 layer_out 是 BF16，h 和 H 是 FP32
layer_out_triton = layer_out_raw.clone().to(dtype).requires_grad_(True)
x_expanded_triton = x_expanded_raw.clone().to(dtype).requires_grad_(True)
h_post_triton = h_post_raw_raw.clone().requires_grad_(True)
H_res_triton = H_res_raw.clone().requires_grad_(True)

# Native 版本的输入 (克隆一份以防梯度污染)
layer_out_native = layer_out_raw.clone().to(dtype).requires_grad_(True)
x_expanded_native = x_expanded_raw.clone().to(dtype).requires_grad_(True)
h_post_native = h_post_raw_raw.clone().requires_grad_(True)
H_res_native = H_res_raw.clone().requires_grad_(True)


# ------------------------------------------------------------------
# 2. 工具函数：数值比较
# ------------------------------------------------------------------
def test_is_close(name, x1, x2, atol=1e-3, rtol=1e-2):
    # 转为 numpy 比较
    x1_np = x1.detach().cpu().float().numpy()
    x2_np = x2.detach().cpu().float().numpy()

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
# 3. 前向测试 (Forward)
# ------------------------------------------------------------------
print("--- 开始前向测试 ---")

# Triton 前向
output_triton = triton_mhc_op(
    layer_out=layer_out_triton,
    x_expanded=x_expanded_triton,
    h_post_raw=h_post_triton,
    H_res=H_res_triton,
)

# Native 前向
output_native = native_mhc_op(
    layer_out=layer_out_native,
    x_expanded=x_expanded_native,
    h_post_raw=h_post_native,
    H_res=H_res_native,
)

test_is_close("Forward Output", output_native, output_triton, atol=1e-2, rtol=1e-2)

# ------------------------------------------------------------------
# 4. 反向测试 (Backward)
# ------------------------------------------------------------------
print("\n--- 开始反向测试 ---")

# 构造一个标量 Loss 进行反向传播
# 使用 mean() 的平方作为一个简单的 loss
loss_triton = (output_triton.float() ** 2).mean()
loss_triton.backward()

loss_native = (output_native.float() ** 2).mean()
loss_native.backward()

# 梯度列表
grad_names = ["layer_out", "x_expanded", "h_post_raw", "H_res"]
triton_grads = [
    layer_out_triton.grad,
    x_expanded_triton.grad,
    h_post_triton.grad,
    H_res_triton.grad,
]
native_grads = [
    layer_out_native.grad,
    x_expanded_native.grad,
    h_post_native.grad,
    H_res_native.grad,
]

for name, g_native, g_triton in zip(grad_names, native_grads, triton_grads):
    if g_native is None or g_triton is None:
        print(f"❌ {name} 梯度不存在")
        continue
    test_is_close(f"Gradient: {name}", g_native, g_triton, atol=1e-2, rtol=1e-2)

print("\n🎉🎉 MHC 全融合算子数值校验结束 🎉🎉")

import time

# ------------------------------------------------------------------
# 5. 性能基准测试 (Benchmark)
# ------------------------------------------------------------------
print("\n" + "=" * 40)
print("🚀 开始性能基准测试 (Speed Benchmark)")
print("=" * 40)

# 定义 Benchmark 配置
n_warmup = 10
n_repeat = 100

# 准备 torch.compile 版本
# mode="reduce-overhead" 适合小算子，"max-autotune" 适合大计算量
# 这里我们用默认或 reduce-overhead 来公平对比
try:
    compiled_mhc_op = torch.compile(native_mhc_op)
except Exception as e:
    print(f"⚠️ torch.compile 不可用: {e}")
    compiled_mhc_op = None

def benchmark_forward(op_func, name, l, x, h, H):
    # 预热
    for _ in range(n_warmup):
        _ = op_func(l, x, h, H)
    torch.cuda.synchronize()

    # 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_repeat):
        _ = op_func(l, x, h, H)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) / n_repeat
    print(f"👉 {name:15s} Forward: {elapsed_time_ms:.4f} ms")
    return elapsed_time_ms

def benchmark_backward(op_func, name, l, x, h, H):
    # 构造 inputs 并设置 grad
    inputs = [t.clone().detach().requires_grad_(True) for t in [l, x, h, H]]
    
    # 预热
    for _ in range(n_warmup):
        out = op_func(*inputs)
        loss = (out.float()**2).mean()
        # 清空梯度
        for t in inputs: 
            if t.grad is not None: t.grad = None
        loss.backward()
    torch.cuda.synchronize()

    # 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_repeat):
        out = op_func(*inputs)
        loss = (out.float()**2).mean()
        # 模拟真实训练：清空梯度 -> 反向传播
        for t in inputs: 
            if t.grad is not None: t.grad = None
        loss.backward()
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) / n_repeat
    print(f"👉 {name:15s} Backward (includes Forward): {elapsed_time_ms:.4f} ms")
    return elapsed_time_ms

# ------------------------------------------------------------------
# 运行测试
# ------------------------------------------------------------------
# 使用同样形状的数据进行测试
# 增加一些 Batch Size 模拟真实高负载场景 (可选)
# B_bench, T_bench = 8, 128 
B_bench, T_bench = B, T 

print(f"配置: B={B_bench}, T={T_bench}, n={n}, C={C}, dtype={dtype}")

l_in = torch.randn(B_bench, T_bench, C, device=device, dtype=dtype)
x_in = torch.randn(B_bench, T_bench, n, C, device=device, dtype=dtype)
h_in = torch.randn(B_bench, T_bench, n, device=device, dtype=torch.float32)
H_in = torch.randn(B_bench, T_bench, n, n, device=device, dtype=torch.float32)

print("\n--- Forward Speed ---")
t_native_fwd = benchmark_forward(native_mhc_op, "Native (Eager)", l_in, x_in, h_in, H_in)
t_triton_fwd = benchmark_forward(triton_mhc_op, "Custom Triton", l_in, x_in, h_in, H_in)
if compiled_mhc_op:
    t_compile_fwd = benchmark_forward(compiled_mhc_op, "torch.compile", l_in, x_in, h_in, H_in)

print("\n--- Backward Speed (Forward + Backward) ---")
t_native_bwd = benchmark_backward(native_mhc_op, "Native (Eager)", l_in, x_in, h_in, H_in)
t_triton_bwd = benchmark_backward(triton_mhc_op, "Custom Triton", l_in, x_in, h_in, H_in)
if compiled_mhc_op:
    t_compile_bwd = benchmark_backward(compiled_mhc_op, "torch.compile", l_in, x_in, h_in, H_in)

# ------------------------------------------------------------------
# 总结分析
# ------------------------------------------------------------------
print("\n" + "=" * 40)
print("📊 性能提升总结 (Speedup)")
print("=" * 40)

def print_speedup(base_name, base_time, target_name, target_time):
    speedup = base_time / target_time
    print(f"{target_name} vs {base_name}: {speedup:.2f}x faster")

print(">>> Forward:")
print_speedup("Native", t_native_fwd, "Triton", t_triton_fwd)
if compiled_mhc_op:
    print_speedup("Compile", t_compile_fwd, "Triton", t_triton_fwd)

print("\n>>> Backward:")
print_speedup("Native", t_native_bwd, "Triton", t_triton_bwd)
if compiled_mhc_op:
    print_speedup("Compile", t_compile_bwd, "Triton", t_triton_bwd)

print("\n🎉🎉🎉🎉 所有测试结束 🎉🎉🎉🎉")