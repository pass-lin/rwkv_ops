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
T = 128
B = 2
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
np.testing.assert_allclose(
    ops.convert_to_numpy(native_out.float()),
    ops.convert_to_numpy(cuda_out.float()),
    atol=1e-3,
    rtol=1e-2,
)
np.testing.assert_allclose(
    ops.convert_to_numpy(native_state.float()),
    ops.convert_to_numpy(cuda_state.float()),
    atol=1e-5,
    rtol=1e-5,
)
print("✅ 前向输出一致")

loss_native = (native_out.mean(1).float() @ native_state.mean(1)).mean() ** 2
loss_native.backward()

loss_cuda = (cuda_out.mean(1).float() @ cuda_state.mean(1)).mean() ** 2
loss_cuda.backward()
# ------------------------------------------------------------------
# 5. 梯度比较
# -----------------------------------w -------------------------------
grad_names = ["r", "k", "v", "a", "b", "w", "h0"]
cuda_grads = [t.grad.float() for t in [r, k, v, a, b, w, h0]]
native_grads = [t.grad.float() for t in [r_n, k_n, v_n, a_n, b_n, w_n, h0_n]]

for name, g_cuda, g_native in zip(grad_names, cuda_grads, native_grads):
    try:
        np.testing.assert_allclose(
            ops.convert_to_numpy(g_native),
            ops.convert_to_numpy(g_cuda),
            atol=2e-2,
            rtol=1e-2,
            err_msg=f"梯度不一致: {name}",
        )
        print(f"✅ {name} 梯度一致")
    except AssertionError as e:
        print(f"❌ {name} 梯度不一致")
        print(e)
        break
    unequal_num = int(ops.sum(g_native - g_cuda))
    all_data_num = int(np.cumprod(g_native.shape)[-1])
    print(
        f"{name} 梯度不一致的元素个数: {unequal_num}, 共计元素个数: {all_data_num},不同的百分比率: {unequal_num / all_data_num:.2%}"
    )
