import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERNEL_TYPE"] = "cuda"

import numpy as np
import jax.numpy as jnp
from keras import ops
from jax import grad

# ------------------------------------------------------------------
# 1. 构造输入
# ------------------------------------------------------------------
T = 128
B = 2
H = 6
K = 64
inputs = [np.random.randn(B, T, H, K) for _ in range(30)]
jax_inputs = [jnp.asarray(t, "bfloat16") for t in inputs]


def normalize(
    z,
    p=2,
    dim=-1,
    eps: float = 1e-12,
):
    # F.normalize like api
    denom = ops.norm(z, ord=p, axis=dim, keepdims=True)
    denom = ops.maximum(denom, 1e-12)
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
# 2. CUDA 版本前向 + 反向
# ------------------------------------------------------------------
from rwkv_ops import rwkv7_op


cuda_out, cuda_state = rwkv7_op(
    r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0, output_final_state=True
)

r_n = ops.copy(r)
k_n = ops.copy(k)
v_n = ops.copy(v)
a_n = ops.copy(a)
b_n = ops.copy(b)
w_n = ops.copy(w)
h0_n = ops.copy(h0)

from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

native_out, native_state = generalized_delta_rule(
    r=r_n, k=k_n, v=v_n, a=a_n, b=b_n, w=w_n, initial_state=h0_n
)


def test_is_close(name, x1, x2, atol=2.5e-2, rtol=1e-3):
    if np.sum(np.abs(x1 - x2)) < 1e-4:
        print(f"✅✅{name} 输出结果完全一致✅✅")
        return
    try:
        np.testing.assert_allclose(
            ops.convert_to_numpy(ops.cast(x1, "float32")),
            ops.convert_to_numpy(ops.cast(x2, "float32")),
            atol=atol,
            rtol=rtol,
        )
        print(f"✅ {name} 梯度一致")
    except AssertionError as e:
        print(f"❌ {name} 梯度不一致")
        print(e)


test_is_close("fwd_pred", native_out, cuda_out, atol=1e-5, rtol=1e-2)
test_is_close("fwd_state", native_state, cuda_state, atol=1e-5, rtol=1e-3)
print(" 前向测试完毕")


# 定义损失函数
def loss_fn(y, state):
    return (y.mean(1).astype("float32") @ state.mean(1)).mean() ** 2


# ------------------------------------------------------------------
# 3. CUDA 版本 + Native 版本反向传播测试
# ------------------------------------------------------------------
print("\n--- 开始反向传播测试 ---")

# 为了确保梯度计算时输入不被修改，我们使用之前创建的副本
# CUDA 版本的输入
cuda_inputs_for_grad = (w, r, k, v, a, b, h0)
# Native 版本的输入
native_inputs_for_grad = (w_n, r_n, k_n, v_n, a_n, b_n, h0_n)


# 定义一个包装函数，它接收所有参数，计算输出，然后计算损失
# JAX的grad函数需要一个标量输出
def cuda_loss_fn(w, r, k, v, a, b, h0):
    """
    CUDA版本的损失计算函数
    """
    y, state = rwkv7_op(
        r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0, output_final_state=True
    )
    return loss_fn(y, state)


def native_loss_fn(w, r, k, v, a, b, h0):
    """
    Native版本的损失计算函数
    """
    y, state = generalized_delta_rule(r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0)
    return loss_fn(y, state)


# 使用 jax.grad 创建梯度函数
# argnums=range(7) 表示我们想对函数的前7个参数（即所有输入）求梯度
# grad函数返回一个函数，该函数的输出是各个参数的梯度组成的元组
cuda_grad_fn = grad(cuda_loss_fn, argnums=range(7))
native_grad_fn = grad(native_loss_fn, argnums=range(7))

# --- 执行梯度计算 ---
print("正在计算 CUDA 版本的梯度...")
# 调用梯度函数，传入输入参数，得到梯度元组
cuda_grads = cuda_grad_fn(*cuda_inputs_for_grad)

print("正在计算 Native 版本的梯度...")
native_grads = native_grad_fn(*native_inputs_for_grad)

# --- 比较梯度 ---
# 梯度的顺序与 argnums 的顺序一致
grad_names = ["w", "r", "k", "v", "a", "b", "h0"]

print("\n--- 梯度比较结果 ---")
for i, name in enumerate(grad_names):
    # 比较每个参数的梯度
    test_is_close(f"grad_{name}", native_grads[i], cuda_grads[i])

print("\n--- 反向传播测试完毕 ---")
