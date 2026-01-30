import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERNEL_TYPE"] = "cuda"

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
# 2. CUDA 版本前向（无 Mask）
# ------------------------------------------------------------------
from rwkv_ops import rwkv7_op_inference  # 推理专用接口

cuda_out, cuda_state = rwkv7_op_inference(
    r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0, output_final_state=True
)

# ------------------------------------------------------------------
# 3. Native 版本前向（无 Mask）
# ------------------------------------------------------------------
from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

native_out, native_state = generalized_delta_rule(
    r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0
)


# ------------------------------------------------------------------
# 4. 前向结果比较工具
# ------------------------------------------------------------------
def test_is_close(name, x1, x2, atol=5e-3, rtol=1e-3):
    x1 = ops.convert_to_numpy(ops.cast(x1, "float32"))
    x2 = ops.convert_to_numpy(ops.cast(x2, "float32"))

    # NaN 检查
    if np.isnan(x1).sum() == 0 and np.isnan(x2).sum() == 0:
        print(f"✅ {name} 无 NaN")
    else:
        print(
            f"❌ {name} 存在 NaN! (CUDA: {np.isnan(x1).sum()}, Native: {np.isnan(x2).sum()})"
        )
        return False

    # 完全一致检查（快速路径）
    if np.abs(x1 - x2).max() < 1e-6:
        print(f"✅ {name} 数值完全一致 (Max Diff < 1e-6)")
        return True

    # 阈值检查
    try:
        np.testing.assert_allclose(x1, x2, atol=atol, rtol=rtol)
        max_diff = np.abs(x1 - x2).max()
        print(f"✅ {name} 一致 (Max Diff: {max_diff:.6e})")
        return True
    except AssertionError as e:
        max_diff = np.abs(x1 - x2).max()
        print(f"❌ {name} 不一致! Max Diff: {max_diff:.6e}")
        print(str(e)[:200] + "...")
        return False


# ------------------------------------------------------------------
# 5. 无 Mask 基础测试
# ------------------------------------------------------------------
print("=" * 60)
print("无 Mask 推理测试")
print("=" * 60)
test_is_close("fwd_pred", native_out, cuda_out, atol=1e-5, rtol=1e-2)
test_is_close("fwd_state", native_state, cuda_state, atol=1e-5, rtol=1e-3)
print("🎉 无 Mask 测试完成\n")

# ------------------------------------------------------------------
# 6. 带 Mask 推理测试
# ------------------------------------------------------------------
print("=" * 60)
print("带 Mask 推理测试")
print("=" * 60)

# 构造随机 Mask（30% 冻结）
mask = torch.ones(B, T, device="cuda", dtype=torch.float32)
freeze_indices = torch.rand(B, T) < 0.3
mask[freeze_indices] = 0.0
print(f"Mask 冻结比例: {(mask == 0).float().mean().item():.2%}")

# CUDA 带 Mask 推理
cuda_out_mask, cuda_state_mask = rwkv7_op_inference(
    r=r,
    k=k,
    v=v,
    a=a,
    b=b,
    w=w,
    mask=mask,  # 传入 mask
    initial_state=h0,
    output_final_state=True,
)

# Native 带 Mask 推理
native_out_mask, native_state_mask = generalized_delta_rule(
    r=r,
    k=k,
    v=v,
    a=a,
    b=b,
    w=w,
    mask=mask,  # 传入 mask
    initial_state=h0,
)

# 对比结果
print("\n>>> 随机 Mask 前向对比")
test_is_close("fwd_pred_mask", native_out_mask, cuda_out_mask, atol=1e-5, rtol=1e-2)
test_is_close(
    "fwd_state_mask", native_state_mask, cuda_state_mask, atol=1e-5, rtol=1e-3
)

# ------------------------------------------------------------------
# 7. Mask 物理一致性验证
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Mask 物理一致性验证")
print("=" * 60)

# 测试 1: 全 0 Mask，状态应该完全不变
print("\n[测试 1] 全 0 Mask（状态冻结）")
mask_all_zero = torch.zeros(B, T, device="cuda", dtype=torch.float32)
h0_frozen = h0.clone()

out_frozen, state_frozen = rwkv7_op_inference(
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
if diff < 1e-5:
    print(f"✅ 状态冻结验证通过 (Max Diff: {diff:.2e})")
else:
    print(f"❌ 状态被错误改变 (Max Diff: {diff:.2e})")

# 测试 2: 全 1 Mask，应与无 Mask 版本完全一致
print("\n[测试 2] 全 1 Mask（与无 Mask 等价）")
mask_all_one = torch.ones(B, T, device="cuda", dtype=torch.float32)

out_one, state_one = rwkv7_op_inference(
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

pred_diff = (out_one - cuda_out).abs().max().item()
state_diff = (state_one - cuda_state).abs().max().item()

if pred_diff < 1e-5 and state_diff < 1e-5:
    print(f"✅ 全 1 Mask 与无 Mask 等价")
    print(f"   输出差异: {pred_diff:.2e}, 状态差异: {state_diff:.2e}")
else:
    print(f"⚠️ 存在微小差异 (输出: {pred_diff:.2e}, 状态: {state_diff:.2e})")

# 测试 3: 边界情况 - 只在最后一个时间步更新
print("\n[测试 3] 仅最后一帧更新（前 T-1 帧冻结）")
mask_last = torch.zeros(B, T, device="cuda", dtype=torch.float32)
mask_last[:, -1] = 1.0  # 只有最后一步更新

out_last, state_last = rwkv7_op_inference(
    r=r,
    k=k,
    v=v,
    a=a,
    b=b,
    w=w,
    mask=mask_last,
    initial_state=h0,
    output_final_state=True,
)

# 验证：最终状态应该只包含最后一步的信息（与 h0 差异不应为零，但也不应过大）
state_change = (state_last - h0).abs().mean().item()
print(f"   状态平均变化量: {state_change:.4f} (应 > 0 且合理)")

print("\n🎉🎉🎉 全部推理测试完成 🎉🎉🎉")
