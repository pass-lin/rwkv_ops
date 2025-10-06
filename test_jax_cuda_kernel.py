import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERNEL_TYPE"] = "cuda"

import numpy as np
import jax.numpy as jnp
from keras import ops

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


def test_is_close(name, x1, x2):
    try:
        np.testing.assert_allclose(
            ops.convert_to_numpy(ops.cast(x1, "float32")),
            ops.convert_to_numpy(ops.cast(x2, "float32")),
            atol=1e-3,
            rtol=1e-2,
        )
        print(f"✅ {name} 梯度一致")
    except AssertionError as e:
        print(f"❌ {name} 梯度不一致")
        print(e)
    unequal_num = int(ops.sum(x1 - x2))
    all_data_num = int(np.cumprod(x1.shape)[-1])
    print(
        f"{name} 梯度不一致的元素个数: {unequal_num}, 共计元素个数: {all_data_num},不同的百分比率: {unequal_num / all_data_num:.2%}"
    )


test_is_close("fwd_pred", native_out, cuda_out)
test_is_close("fwd_state", native_state, cuda_state)
print(" 前向测试完毕")
