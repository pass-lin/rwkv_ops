import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERNEL_TYPE"] = "cuda"

import numpy as np
import jax.numpy as jnp
from keras import ops
from jax import grad

# ------------------------------------------------------------------
# 1. 构造输入
# ------------------------------------------------------------------
T = 512
B = 5
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


mask = ops.concatenate(
    [ops.zeros((B, T // 2, 1, 1)), ops.ones((B, T // 2, 1, 1))], axis=1
)
mask = ops.cast(mask, "bfloat16")

a = -normalize(jax_inputs[3], dim=-1, p=2.0) * mask
b = normalize(jax_inputs[3], dim=-1, p=2.0) * mask

w = jax_inputs[4] * mask  # decay / gate
r = jax_inputs[0] * mask  # receptance
k = jax_inputs[1] * mask
v = jax_inputs[2] * mask
w = -ops.softplus(w) - 0.5
w = ops.where(mask, w, -1e9)
h0 = jnp.asarray(np.random.randn(B, H, K, K), "float32")
# ------------------------------------------------------------------
# 2. CUDA 版本前向 + 反向
# ------------------------------------------------------------------
from rwkv_ops import rwkv7_op_inference


cuda_out, cuda_state = rwkv7_op_inference(
    r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0
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


def test_is_close(name, x1, x2, atol=5e-3, rtol=1e-3):
    x1 = ops.convert_to_numpy(ops.cast(x1, "float32"))
    x2 = ops.convert_to_numpy(ops.cast(x2, "float32"))
    if np.sum(np.isnan(x1)) == 0 and np.sum(np.isnan(x2)) == 0:
        print(f"✅✅{name} 不存在nan✅✅")
    else:
        print(f"❌❌{name} 你妈的有nan❌❌")
    if np.sum(np.abs(x1 - x2)) < 1e-4:
        print(f"✅✅{name} 输出结果完全一致✅✅")
        return
    try:
        np.testing.assert_allclose(
            x1,
            x2,
            atol=atol,
            rtol=rtol,
        )
        print(f"✅ {name} 梯度一致")
    except AssertionError as e:
        print(f"❌ {name} 梯度不一致")
        print(e)


test_is_close("fwd_pred", native_out, cuda_out, atol=1e-5, rtol=1e-2)
test_is_close("fwd_state", native_state, cuda_state, atol=1e-5, rtol=1e-3)
print("🎉🎉🎉🎉test_script/test_jax_cuda_kernel_inference.py测试结束🎉🎉🎉🎉")
