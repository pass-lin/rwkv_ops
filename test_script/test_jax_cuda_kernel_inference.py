import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERNEL_TYPE"] = "cuda"

import numpy as np
import jax.numpy as jnp
from keras import ops

# ------------------------------------------------------------------
# 1. 构造输入
# ------------------------------------------------------------------
T = 128  # 可以根据需要调整，推理可以用较大值
B = 5
H = 6
K = 64
inputs = [np.random.randn(B, T, H, K) for _ in range(30)]
jax_inputs = [jnp.asarray(t, "bfloat16") for t in inputs]


def normalize(z, p=2, dim=-1, eps: float = 1e-12):
    # F.normalize like api
    denom = ops.norm(z, ord=p, axis=dim, keepdims=True)
    denom = ops.maximum(denom, eps)
    return z / denom


a = -normalize(jax_inputs[3], dim=-1, p=2.0)
b = normalize(jax_inputs[3], dim=-1, p=2.0)

w = jax_inputs[4]  # decay / gate
r = jax_inputs[0]  # receptance
k = jax_inputs[1]
v = jax_inputs[2]
w = -ops.softplus(w) - 0.5
h0 = jnp.asarray(np.random.randn(B, H, K, K), "float32")


# ------------------------------------------------------------------
# 2. 辅助函数
# ------------------------------------------------------------------
def test_is_close(name, x1, x2, atol=5e-3, rtol=1e-3):
    x1 = ops.convert_to_numpy(ops.cast(x1, "float32"))
    x2 = ops.convert_to_numpy(ops.cast(x2, "float32"))

    # NaN 检查
    if np.sum(np.isnan(x1)) == 0 and np.sum(np.isnan(x2)) == 0:
        print(f"✅ {name} 无 NaN")
    else:
        print(f"❌ {name} 存在 NaN!")
        print(f"   x1 NaN 数量: {np.sum(np.isnan(x1))}")
        print(f"   x2 NaN 数量: {np.sum(np.isnan(x2))}")
        return False

    # 完全一致检查（快速路径）
    if np.allclose(x1, x2, atol=1e-6):
        print(f"✅ {name} 数值完全一致")
        return True

    # 阈值检查
    try:
        np.testing.assert_allclose(x1, x2, atol=atol, rtol=rtol)
        max_diff = np.abs(x1 - x2).max()
        print(f"✅ {name} 一致 (Max Diff: {max_diff:.6e})")
        return True
    except AssertionError:
        max_diff = np.abs(x1 - x2).max()
        print(f"❌ {name} 不一致! Max Diff: {max_diff:.6e}")
        return False


# ------------------------------------------------------------------
# 3. 导入模块
# ------------------------------------------------------------------
from rwkv_ops import rwkv7_op_inference
from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

print("=" * 60)
print("无 Mask 推理测试")
print("=" * 60)

# ------------------------------------------------------------------
# 3.1 无 Mask 前向测试
# ------------------------------------------------------------------
cuda_out, cuda_state = rwkv7_op_inference(
    r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0
)

# Native 版本（detach 副本）
r_n = ops.copy(r)
k_n = ops.copy(k)
v_n = ops.copy(v)
a_n = ops.copy(a)
b_n = ops.copy(b)
w_n = ops.copy(w)
h0_n = ops.copy(h0)

native_out, native_state = generalized_delta_rule(
    r=r_n, k=k_n, v=v_n, a=a_n, b=b_n, w=w_n, initial_state=h0_n
)

test_is_close("fwd_pred", native_out, cuda_out, atol=1e-5, rtol=1e-2)
test_is_close("fwd_state", native_state, cuda_state, atol=1e-5, rtol=1e-3)
print("🎉 无 Mask 推理测试完成")

# ------------------------------------------------------------------
# 4. 带 Mask 推理测试
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("带 Mask 推理测试")
print("=" * 60)

# 构造随机 Mask（30% 冻结）
mask_np = np.ones((B, T), dtype=np.float32)
freeze_indices = np.random.rand(B, T) < 0.3
mask_np[freeze_indices] = 0.0
mask = jnp.asarray(mask_np)

print(f"Mask 冻结比例: {(mask == 0).mean():.2%}")

# 为了保证数值稳定性，重新生成一份输入（避免之前的影响）
inputs_mask = [np.random.randn(B, T, H, K) for _ in range(30)]
jax_inputs_mask = [jnp.asarray(t, "bfloat16") for t in inputs_mask]

a_m = -normalize(jax_inputs_mask[3], dim=-1, p=2.0)
b_m = normalize(jax_inputs_mask[3], dim=-1, p=2.0)
w_m = jax_inputs_mask[4]
r_m = jax_inputs_mask[0]
k_m = jax_inputs_mask[1]
v_m = jax_inputs_mask[2]
w_m = -ops.softplus(w_m) - 0.5
h0_m = jnp.asarray(np.random.randn(B, H, K, K), "float32")

# Native 副本
r_m_n = ops.copy(r_m)
k_m_n = ops.copy(k_m)
v_m_n = ops.copy(v_m)
a_m_n = ops.copy(a_m)
b_m_n = ops.copy(b_m)
w_m_n = ops.copy(w_m)
h0_m_n = ops.copy(h0_m)

# ------------------------------------------------------------------
# 4.1 带 Mask 前向测试
# ------------------------------------------------------------------
cuda_out_mask, cuda_state_mask = rwkv7_op_inference(
    r=r_m, k=k_m, v=v_m, a=a_m, b=b_m, w=w_m, initial_state=h0_m, mask=mask
)

native_out_mask, native_state_mask = generalized_delta_rule(
    r=r_m_n,
    k=k_m_n,
    v=v_m_n,
    a=a_m_n,
    b=b_m_n,
    w=w_m_n,
    initial_state=h0_m_n,
    mask=mask,
)

print("\n>>> 随机 Mask 前向对比")
test_is_close("fwd_pred_mask", native_out_mask, cuda_out_mask, atol=1e-5, rtol=1e-2)
test_is_close(
    "fwd_state_mask", native_state_mask, cuda_state_mask, atol=1e-5, rtol=1e-3
)

# ------------------------------------------------------------------
# 5. Mask 物理一致性验证
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Mask 物理一致性验证")
print("=" * 60)

# 测试 1: 全 0 Mask，状态应该完全不变
print("\n[测试 1] 全 0 Mask（状态冻结）")
mask_all_zero = jnp.zeros((B, T), jnp.float32)
h0_frozen = h0_m.copy()

out_frozen, state_frozen = rwkv7_op_inference(
    r=r_m,
    k=k_m,
    v=v_m,
    a=a_m,
    b=b_m,
    w=w_m,
    initial_state=h0_frozen,
    mask=mask_all_zero,
)

diff = jnp.abs(state_frozen - h0_frozen).max()
if diff < 1e-5:
    print(f"✅ 状态冻结验证通过 (Max Diff: {diff:.2e})")
else:
    print(f"❌ 状态被错误改变 (Max Diff: {diff:.2e})")

# 测试 2: 全 1 Mask，应与无 Mask 版本一致（使用相同输入）
print("\n[测试 2] 全 1 Mask（与无 Mask 等价）")
mask_all_one = jnp.ones((B, T), jnp.float32)

# 使用无 mask 版本的输入重新测试
out_one_mask, state_one_mask = rwkv7_op_inference(
    r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0, mask=mask_all_one
)

# 注意：这里与之前的 cuda_out/cuda_state 比较（无 mask 版本）
pred_diff = jnp.abs(out_one_mask - cuda_out).max()
state_diff = jnp.abs(state_one_mask - cuda_state).max()

if pred_diff < 1e-5 and state_diff < 1e-5:
    print("✅ 全 1 Mask 与无 Mask 等价")

else:
    print(f"⚠️ 存在微小差异 (输出: {pred_diff:.2e}, 状态: {state_diff:.2e})")

# 测试 3: 边界情况 - 只在最后一个时间步更新
print("\n[测试 3] 仅最后一帧更新（前 T-1 帧冻结）")
mask_last = np.zeros((B, T), dtype=np.float32)
mask_last[:, -1] = 1.0  # 只有最后一步更新
mask_last = jnp.asarray(mask_last)

out_last, state_last = rwkv7_op_inference(
    r=r_m, k=k_m, v=v_m, a=a_m, b=b_m, w=w_m, mask=mask_last, initial_state=h0_m
)

# 验证：最终状态应该与初始状态有差异（因为最后一步更新了）
state_change = jnp.abs(state_last - h0_m).mean()
if state_change > 0:
    print(f"✅ 最后一帧更新验证通过 (状态平均变化: {state_change:.4f})")
else:
    print("⚠️ 状态未变化，可能存在问题")

print("\n" + "=" * 60)
print("🎉🎉🎉 全部 JAX 推理测试完成 🎉🎉🎉")
print("=" * 60)
