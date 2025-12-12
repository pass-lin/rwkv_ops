import os
import argparse

parser = argparse.ArgumentParser(
    description="Test RWKV7 Op with different backends and kernel types."
)
parser.add_argument(
    "--backend",
    type=str,
    choices=["torch", "jax", "numpy", "tensorflow", "openvino"],
    required=True,
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = args.backend
os.environ["KERNEL_TYPE"] = "cuda"

import numpy as np
from keras import ops

# ------------------------------------------------------------------
# 1. 构造输入
# ------------------------------------------------------------------
T = 512
B = 5
H = 6
K = 64
inputs = [np.random.randn(B, T, H, K) for _ in range(30)]
tf_inputs = [ops.convert_to_tensor(t, "bfloat16") for t in inputs]


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

a = -normalize(tf_inputs[3], dim=-1, p=2.0) * mask
b = normalize(tf_inputs[3], dim=-1, p=2.0) * mask

w = tf_inputs[4] * mask  # decay / gate
r = tf_inputs[0] * mask  # receptance
k = tf_inputs[1] * mask
v = tf_inputs[2] * mask
w = -ops.softplus(w) - 0.5
w = ops.where(mask, w, -1e9)
h0 = ops.convert_to_tensor(np.random.randn(B, H, K, K), "float32")
# ------------------------------------------------------------------
# 2. CUDA 版本前向 + 反向
# ------------------------------------------------------------------
from rwkv_ops import rwkv7_op


cuda_out, cuda_state = rwkv7_op(r=r, k=k, v=v, a=a, b=b, w=w, initial_state=h0)

r_n = ops.copy(r)
k_n = ops.copy(k)
v_n = ops.copy(v)
a_n = ops.copy(a)
b_n = ops.copy(b)
w_n = ops.copy(w)
h0_n = ops.copy(h0)


def transpose_head(x, head_first):
    """
    对输入张量进行转置操作。

    参数:
    x: 输入张量。
    head_first: 布尔值，决定是否进行转置。

    返回:
    转置后的张量（如果head_first为True），否则返回原张量。
    """
    x = ops.cast(x, "float32")
    if head_first:
        return ops.transpose(x, (0, 2, 1, 3))
    else:
        return x


def generalized_delta_rule(
    r,
    w,
    k,
    v,
    a,
    b,
    initial_state=None,
    output_final_state: bool = True,
    head_first: bool = False,
):
    """
    实现广义delta规则的函数。

    参数:
    r: 输入张量。
    w: 权重张量。
    k, v, a, b: 其他输入张量。
    initial_state: 初始状态张量。
    output_final_state: 是否输出最终状态。
    head_first: 是否在计算中将head维度放在第一位。

    返回:
    根据output_final_state参数决定是否返回最终状态。
    """
    DTYPE = r.dtype
    B, T, H, N = ops.shape(r)
    r = transpose_head(r, head_first)

    k = transpose_head(k, head_first)

    v = transpose_head(v, head_first)
    a = transpose_head(a, head_first)
    b = transpose_head(b, head_first)
    w = transpose_head(w, head_first)
    w = ops.exp(-ops.exp(w))

    if initial_state is not None:
        state = initial_state
        if ops.shape(state)[0] == 1:
            state = ops.broadcast_to(state, (B, H, N, N))
    else:
        state = ops.zeros((B, H, N, N))
    state = ops.cast(state, "float32")
    out = ops.zeros((B, T, H, N), DTYPE)

    def step(t, inputs):
        """
        执行单个时间步的计算。

        参数:
        t: 当前时间步。
        inputs: 包含当前状态和输出的列表。

        返回:
        更新后的状态和输出。
        """
        state, out = inputs
        kk = ops.reshape(k[:, t, :], (B, H, 1, N))
        rr = ops.reshape(r[:, t, :], (B, H, N, 1))
        vv = ops.reshape(v[:, t, :], (B, H, N, 1))
        aa = ops.reshape(a[:, t, :], (B, H, N, 1))
        bb = ops.reshape(b[:, t, :], (B, H, 1, N))
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        o = ops.cast((state @ rr), out.dtype)
        out = ops.slice_update(out, [0, t, 0, 0], ops.reshape(o, (B, 1, H, N)))
        return [state, out]

    state, out = ops.fori_loop(0, T, step, [state, out])

    if output_final_state:
        return ops.cast(out, DTYPE), state
    return ops.cast(out, DTYPE)


native_out, native_state = generalized_delta_rule(
    r=r_n, k=k_n, v=v_n, a=a_n, b=b_n, w=w_n, initial_state=h0_n
)


def test_is_close(name, x1, x2, atol=2.5e-2, rtol=1e-3):
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
print(f"🎉🎉🎉🎉test_script/test_rwkv7_native.py在{args.backend}后端的测试结束🎉🎉🎉🎉")
