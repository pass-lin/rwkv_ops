import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERNEL_TYPE"] = "triton"

import numpy as np
import torch
from torch.nn import functional as F
from keras import ops

# ------------------------------------------------------------------
# 1. 构造输入
# ------------------------------------------------------------------
T = 1024
B = 5
H = 6
K = 64
inputs = [np.random.randn(B, T, H, K) for _ in range(30)]
torch_inputs = [torch.from_numpy(t).bfloat16().cuda() for t in inputs]

a = -F.normalize(torch_inputs[3], dim=-1, p=2.0)
b = F.normalize(torch_inputs[3], dim=-1, p=2.0)

w = torch_inputs[4]  # decay / gate
r = torch_inputs[0]  # receptance
k = torch_inputs[1]
v = torch_inputs[2]
w = -ops.softplus(w) - 0.5
h0 = torch.from_numpy(np.random.randn(B, H, K, K)).float().cuda()
# ------------------------------------------------------------------
# 2. CUDA 版本前向 + 反向
# ------------------------------------------------------------------
from rwkv_ops import rwkv7_op  # 即 get_generalized_delta_rule 返回的函数

# 要求计算梯度，必须设置 requires_grad
for t in [r, k, v, a, b, w, h0]:
    t.requires_grad_(True)

cuda_out, cuda_state = rwkv7_op(
    r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0, output_final_state=True
)


# ------------------------------------------------------------------
# 3. Native 版本前向 + 反向
# ------------------------------------------------------------------
from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

# 重新构造一份 detach 的新张量，避免梯度混淆
r_n = r.detach().clone().requires_grad_(True)
k_n = k.detach().clone().requires_grad_(True)
v_n = v.detach().clone().requires_grad_(True)
a_n = a.detach().clone().requires_grad_(True)
b_n = b.detach().clone().requires_grad_(True)
w_n = w.detach().clone().requires_grad_(True)
h0_n = h0.detach().clone().requires_grad_(True)

native_out, native_state = generalized_delta_rule(
    r=r_n, k=k_n, v=v_n, a=a_n, b=b_n, w=w_n, initial_state=h0_n
)


# ------------------------------------------------------------------
# 4. 前向结果比较
# ------------------------------------------------------------------
def test_is_close(name, x1, x2, atol=5e-3, rtol=1e-3):
    t1 = ops.convert_to_numpy(ops.cast(x1, "float32"))
    t2 = ops.convert_to_numpy(ops.cast(x2, "float32"))
    error = np.abs(t1 - t2)
    max_diff = error.max()
    avg_error = error.mean()
    rel_error = avg_error / (np.abs(t1).mean() + 1e-9)
    print("-" * 100)
    print(f"[{name}]")
    print(f"  平均绝对误差: {avg_error:.6e}")
    print(f"  平均相对误差: {rel_error:.6e}")

    equal_rate = np.sum(error < 1e-7) / t1.size * 100
    print(f"  {equal_rate:.2f}% 的数据完全一样")
    try:
        np.testing.assert_allclose(t1, t2, atol=atol, rtol=rtol)
        print(f"✅ {name} 一致 (Max Diff: {max_diff:.6e})")
        return True
    except AssertionError as e:
        print(f"❌ {name} 不一致! Max Diff: {max_diff:.6e}")
        print(e)
        return False


test_is_close("fwd_pred", native_out, cuda_out, atol=1e-5, rtol=1e-2)
test_is_close("fwd_state", native_state, cuda_state, atol=1e-5, rtol=1e-3)
print("前向测试完毕")

loss_native = ((native_out).pow(2).mean() - (native_state).pow(2).mean()).abs()
loss_native.backward()

loss_cuda = ((cuda_out).pow(2).mean() - (cuda_state).pow(2).mean()).abs()
loss_cuda.backward()
# ------------------------------------------------------------------
# 5. 梯度比较
# -----------------------------------w -------------------------------
grad_names = ["r", "k", "v", "a", "b", "w", "h0"]
cuda_grads = [t.grad.float() for t in [r, k, v, a, b, w, h0]]
native_grads = [t.grad.float() for t in [r_n, k_n, v_n, a_n, b_n, w_n, h0_n]]

for name, g_cuda, g_native in zip(grad_names, cuda_grads, native_grads):
    test_is_close(name, g_cuda, g_native, atol=7e-3)
print("🎉🎉🎉🎉test_script/test_torch_cuda_kernel.py测试结束🎉🎉🎉🎉")

# ------------------------------------------------------------------
# 6. 带 Mask 版本测试
# ------------------------------------------------------------------
print("\n" + "=" * 50 + " 带 Mask 版本测试 " + "=" * 50)

# 构造 Mask：随机混合 0 和 1，模拟部分更新场景
# 形状 [B, T]，值 0.0 或 1.0
mask = torch.ones(B, T, device="cuda", dtype=torch.float32)
# 随机冻结某些位置（例如 30% 的时间步不更新状态）
freeze_indices = torch.rand(B, T) < 0.3
mask[freeze_indices] = 0.0

print(f"Mask 冻结比例: {(mask == 0).float().mean().item():.2%}")

# 重新构造输入（避免与之前测试的梯度混淆）
r_m = r.detach().clone().requires_grad_(True)
k_m = k.detach().clone().requires_grad_(True)
v_m = v.detach().clone().requires_grad_(True)
a_m = a.detach().clone().requires_grad_(True)
b_m = b.detach().clone().requires_grad_(True)
w_m = w.detach().clone().requires_grad_(True)
h0_m = h0.detach().clone().requires_grad_(True)

r_m_n = r.detach().clone().requires_grad_(True)
k_m_n = k.detach().clone().requires_grad_(True)
v_m_n = v.detach().clone().requires_grad_(True)
a_m_n = a.detach().clone().requires_grad_(True)
b_m_n = b.detach().clone().requires_grad_(True)
w_m_n = w.detach().clone().requires_grad_(True)
h0_m_n = h0.detach().clone().requires_grad_(True)
mask[:, -5:] = 0
# ------------------------------------------------------------------
# 7. CUDA 带 Mask 前向 + 反向
# ------------------------------------------------------------------
cuda_out_mask, cuda_state_mask = rwkv7_op(
    r=r_m,
    k=k_m,
    v=v_m,
    a=a_m,
    b=b_m,
    w=w_m,
    mask=mask,  # 传入 mask
    initial_state=h0_m,
    output_final_state=True,
)

# ------------------------------------------------------------------
# 8. Native 带 Mask 前向 + 反向
# ------------------------------------------------------------------


native_out_mask, native_state_mask = generalized_delta_rule(
    r=r_m_n,
    k=k_m_n,
    v=v_m_n,
    a=a_m_n,
    b=b_m_n,
    w=w_m_n,
    mask=mask,  # 传入 mask
    initial_state=h0_m_n,
)

# ------------------------------------------------------------------
# 9. 带 Mask 前向结果比较
# ------------------------------------------------------------------
print("\n>>> 带 Mask 前向对比")
test_is_close("fwd_pred_mask", native_out_mask, cuda_out_mask, atol=1e-5, rtol=1e-2)
test_is_close(
    "fwd_state_mask", native_state_mask, cuda_state_mask, atol=1e-5, rtol=1e-3
)

# ------------------------------------------------------------------
# 10. 带 Mask 梯度比较
# ------------------------------------------------------------------
loss_cuda_mask = (cuda_out_mask.pow(2).mean() - cuda_state_mask.pow(2).mean()).abs()
loss_cuda_mask.backward()

loss_native_mask = (
    native_out_mask.pow(2).mean() - native_state_mask.pow(2).mean()
).abs()
loss_native_mask.backward()

print("\n>>> 带 Mask 梯度对比")
grad_names = ["r", "k", "v", "a", "b", "w", "h0"]
cuda_grads_mask = [t.grad.float() for t in [r_m, k_m, v_m, a_m, b_m, w_m, h0_m]]
native_grads_mask = [
    t.grad.float() for t in [r_m_n, k_m_n, v_m_n, a_m_n, b_m_n, w_m_n, h0_m_n]
]

for name, g_cuda, g_native in zip(grad_names, cuda_grads_mask, native_grads_mask):
    test_is_close(f"{name}_mask", g_cuda, g_native, atol=7e-3)

# ------------------------------------------------------------------
# 11. Mask 物理一致性验证（关键测试）
# ------------------------------------------------------------------
print("\n>>> Mask 物理一致性验证")

# 测试：全 0 Mask 时状态应该完全不变
mask_all_zero = torch.zeros(B, T, device="cuda", dtype=torch.float32)
h0_frozen = h0.detach().clone()

with torch.no_grad():
    out_frozen, state_frozen = rwkv7_op(
        r=r,
        k=k,
        v=v,
        a=a,
        b=b,
        w=w,
        mask=mask_all_zero,
        initial_state=h0_frozen,
        output_final_state=True,
    )

    diff = (state_frozen - h0_frozen).abs().max().item()
    if diff < 1e-6:
        print(f"✅ 全 0 Mask 状态冻结验证通过 (Max Diff: {diff:.2e})")
    else:
        print(f"❌ 全 0 Mask 状态错误改变 (Max Diff: {diff:.2e})")

# 测试：全 1 Mask 时应该与无 Mask 版本等价
mask_all_one = torch.ones(B, T, device="cuda", dtype=torch.float32)

with torch.no_grad():
    out_one_mask, state_one_mask = rwkv7_op(
        r=r,
        k=k,
        v=v,
        a=a,
        b=b,
        w=w,
        mask=mask_all_one,
        initial_state=h0,
        output_final_state=True,
    )

    # 与之前无 mask 的 cuda_out/cuda_state 比较
    pred_diff = (out_one_mask - cuda_out).abs().max().item()
    state_diff = (state_one_mask - cuda_state).abs().max().item()

    if pred_diff < 1e-5 and state_diff < 1e-5:
        print(
            f"✅ 全 1 Mask 与无 Mask 等价 (Pred Diff: {pred_diff:.2e}, State Diff: {state_diff:.2e})"
        )
    else:
        print(
            f"⚠️ 全 1 Mask 与无 Mask 存在差异 (Pred Diff: {pred_diff:.2e}, State Diff: {state_diff:.2e})"
        )

print("\n" + "=" * 50 + " Mask 测试结束 " + "=" * 50)
