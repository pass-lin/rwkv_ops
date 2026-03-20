import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERNEL_TYPE"] = "cuda"

import numpy as np
import jax.numpy as jnp
from keras import ops
from jax import grad

# ------------------------------------------------------------------
# 1. 构造输入
# ------------------------------------------------------------------
T = 128  # 减小 T 以便更快测试，如需大 T 可改回 512
B = 5
H = 6
K = 64
inputs = [np.random.randn(B, H, T, K) for _ in range(30)]
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
def test_is_close(name, x1, x2, atol=1e-2, rtol=1e-3):
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


# 定义损失函数
def loss_fn(y, state):
    return jnp.abs((y**2).mean().astype("float32") - (state**2).mean())


# ------------------------------------------------------------------
# 3. 导入模块
# ------------------------------------------------------------------
from rwkv_ops import rwkv7_op
from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

print("=" * 60)
print("无 Mask 版本测试")
print("=" * 60)

# ------------------------------------------------------------------
# 3.1 无 Mask 前向测试
# ------------------------------------------------------------------
cuda_out, cuda_state = rwkv7_op(
    r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0, head_first=True
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
    r=r_n, k=k_n, v=v_n, a=a_n, b=b_n, w=w_n, initial_state=h0_n, head_first=True
)

test_is_close("fwd_pred", native_out, cuda_out, atol=1e-5, rtol=1e-2)
test_is_close("fwd_state", native_state, cuda_state, atol=1e-5, rtol=1e-3)
print("前向测试完毕")

# ------------------------------------------------------------------
# 3.2 无 Mask 反向测试
# ------------------------------------------------------------------
print("\n--- 无 Mask 反向传播测试 ---")


def cuda_loss_fn(w, r, k, v, a, b, h0):
    y, state = rwkv7_op(
        r=r,
        k=k,
        v=v,
        a=a,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
        head_first=True,
    )
    return loss_fn(y, state)


def native_loss_fn(w, r, k, v, a, b, h0):
    y, state = generalized_delta_rule(
        r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0, head_first=True
    )
    return loss_fn(y, state)


cuda_grad_fn = grad(cuda_loss_fn, argnums=range(7))
native_grad_fn = grad(native_loss_fn, argnums=range(7))

cuda_grads = cuda_grad_fn(w, r, k, v, a, b, h0)
native_grads = native_grad_fn(w_n, r_n, k_n, v_n, a_n, b_n, h0_n)

grad_names = ["w", "r", "k", "v", "a", "b", "h0"]
print("\n梯度比较结果:")
for i, name in enumerate(grad_names):
    test_is_close(f"grad_{name}", native_grads[i], cuda_grads[i], atol=7e-3)
print("测试非连续和head_frist的情况完成")

# ------------------------------------------------------------------
# 1. 构造输入
# ------------------------------------------------------------------
T = 128  # 减小 T 以便更快测试，如需大 T 可改回 512
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
def test_is_close(name, x1, x2, atol=1e-2, rtol=1e-3):
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


# 定义损失函数
def loss_fn(y, state):
    return jnp.abs((y**2).mean().astype("float32") - (state**2).mean())


# ------------------------------------------------------------------
# 3. 导入模块
# ------------------------------------------------------------------
from rwkv_ops import rwkv7_op
from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

print("=" * 60)
print("无 Mask 版本测试")
print("=" * 60)

# ------------------------------------------------------------------
# 3.1 无 Mask 前向测试
# ------------------------------------------------------------------
cuda_out, cuda_state = rwkv7_op(r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0)

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
print("前向测试完毕")

# ------------------------------------------------------------------
# 3.2 无 Mask 反向测试
# ------------------------------------------------------------------
print("\n--- 无 Mask 反向传播测试 ---")


def cuda_loss_fn(w, r, k, v, a, b, h0):
    y, state = rwkv7_op(
        r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0, output_final_state=True
    )
    return loss_fn(y, state)


def native_loss_fn(w, r, k, v, a, b, h0):
    y, state = generalized_delta_rule(r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0)
    return loss_fn(y, state)


cuda_grad_fn = grad(cuda_loss_fn, argnums=range(7))
native_grad_fn = grad(native_loss_fn, argnums=range(7))

cuda_grads = cuda_grad_fn(w, r, k, v, a, b, h0)
native_grads = native_grad_fn(w_n, r_n, k_n, v_n, a_n, b_n, h0_n)

grad_names = ["w", "r", "k", "v", "a", "b", "h0"]
print("\n梯度比较结果:")
for i, name in enumerate(grad_names):
    test_is_close(f"grad_{name}", native_grads[i], cuda_grads[i], atol=7e-3)

print("🎉 无 Mask 测试完成")

# ------------------------------------------------------------------
# 4. 带 Mask 版本测试
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("带 Mask 版本测试")
print("=" * 60)

# 构造随机 Mask（30% 冻结）
mask_np = np.ones((B, T), dtype=np.float32)
freeze_indices = np.random.rand(B, T) < 0.3
mask_np[freeze_indices] = 0.0
mask_np[:, -5:] = 0

mask = jnp.asarray(mask_np)

print(f"Mask 冻结比例: {(mask == 0).mean():.2%}")

# 为了保证数值稳定性，重新生成一份输入（避免之前梯度计算的影响）
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
cuda_out_mask, cuda_state_mask = rwkv7_op(
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
# 4.2 带 Mask 反向测试
# ------------------------------------------------------------------
print("\n--- 带 Mask 反向传播测试 ---")


def cuda_loss_fn_mask(w, r, k, v, a, b, h0, mask):
    y, state = rwkv7_op(
        r=r,
        k=k,
        v=v,
        a=a,
        b=b,
        w=w,
        initial_state=h0,
        mask=mask,
        output_final_state=True,
    )
    return loss_fn(y, state)


def native_loss_fn_mask(w, r, k, v, a, b, h0, mask):
    y, state = generalized_delta_rule(
        r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0, mask=mask
    )
    return loss_fn(y, state)


cuda_grad_fn_mask = grad(cuda_loss_fn_mask, argnums=range(7))
native_grad_fn_mask = grad(native_loss_fn_mask, argnums=range(7))

cuda_grads_mask = cuda_grad_fn_mask(w_m, r_m, k_m, v_m, a_m, b_m, h0_m, mask)
native_grads_mask = native_grad_fn_mask(
    w_m_n, r_m_n, k_m_n, v_m_n, a_m_n, b_m_n, h0_m_n, mask
)

print("\nMask 梯度比较结果:")
for i, name in enumerate(grad_names):
    test_is_close(
        f"grad_{name}_mask", native_grads_mask[i], cuda_grads_mask[i], atol=7e-3
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

out_frozen, state_frozen = rwkv7_op(
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
out_one_mask, state_one_mask = rwkv7_op(
    r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0, mask=mask_all_one
)

# 注意：这里与之前的 cuda_out/cuda_state 比较（无 mask 版本）
pred_diff = jnp.abs(out_one_mask - cuda_out).max()
state_diff = jnp.abs(state_one_mask - cuda_state).max()

if pred_diff < 1e-5 and state_diff < 1e-5:
    print("✅ 全 1 Mask 与无 Mask 等价")
else:
    print(f"⚠️ 存在微小差异 (输出: {pred_diff:.2e}, 状态: {state_diff:.2e})")

print("\n" + "=" * 60)
print("🎉🎉🎉 全部 JAX 测试完成 🎉🎉🎉")
print("=" * 60)
