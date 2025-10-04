import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERNEL_TYPE"] = "cuda"
import numpy as np
import torch
from torch.nn import functional as F
from keras import ops

T = 128
B = 2
H = 6
K = 64
inputs = [np.random.randn(B, T, H, K) for _ in range(30)]
torch_inputs = [torch.from_numpy(t).bfloat16().cuda() for t in inputs]

a = -F.normalize(torch_inputs[3], dim=-1, p=2.0)
b = F.normalize(torch_inputs[3], dim=-1, p=2.0)

from rwkv_ops import rwkv7_op

cuda_out, cuda_state = rwkv7_op(
    r=torch_inputs[0],
    k=torch_inputs[1],
    v=torch_inputs[2],
    a=a,
    b=b,
    w=torch_inputs[4],
    initial_state=None,
    output_final_state=True,
)

# 测试一下原生的python实现
from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

native_out, native_state = generalized_delta_rule(
    r=torch_inputs[0],
    k=torch_inputs[1],
    v=torch_inputs[2],
    a=a,
    b=b,
    w=torch_inputs[4],
)
np.testing.assert_allclose(
    ops.convert_to_numpy(native_out.float()),
    ops.convert_to_numpy(cuda_out.float()),
    atol=5e-3,
    rtol=1e-2,
)
np.testing.assert_allclose(
    ops.convert_to_numpy(native_state.float()),
    ops.convert_to_numpy(cuda_state.float()),
    atol=1e-5,
    rtol=1e-5,
)

raise (1)
# ===== 1. 构造标量损失，让 pytorch 自动求 grad =====
loss_cuda = (cuda_out * torch.randn_like(cuda_out)).sum() + (
    cuda_state * torch.randn_like(cuda_state)
).sum()

# 对 6 个输入 tensor 求梯度
torch_inputs_cuda = [
    torch_inputs[0],
    torch_inputs[1],
    torch_inputs[2],
    a,
    b,
    torch_inputs[4],
]
grad_cuda = torch.autograd.grad(
    loss_cuda, torch_inputs_cuda, retain_graph=False, allow_unused=True
)

# ===== 2. 原生实现打开求踪 =====
for t in torch_inputs_cuda:
    t.requires_grad_(True)

native_out, native_state = generalized_delta_rule(
    r=torch_inputs_cuda[0],
    k=torch_inputs_cuda[1],
    v=torch_inputs_cuda[2],
    a=torch_inputs_cuda[3],
    b=torch_inputs_cuda[4],
    w=torch_inputs_cuda[5],
)
loss_native = (native_out * torch.randn_like(native_out)).sum() + (
    native_state * torch.randn_like(native_state)
).sum()

grad_native = torch.autograd.grad(
    loss_native, torch_inputs_cuda, retain_graph=False, allow_unused=True
)

# ===== 3. 梯度数值比对 =====
print(">>> 梯度误差 (CUDA vs Native)")
for i, (gc, gn) in enumerate(zip(grad_cuda, grad_native)):
    gc = gc.float().detach().cpu().numpy()
    gn = gn.float().detach().cpu().numpy()
    err = np.abs(gc - gn).max()
    rel = err / (np.abs(gn).max() + 1e-7)
    print(f"  input[{i}]  max_abs_err={err:.6f}  rel_err={rel:.6f}")
    np.testing.assert_allclose(gc, gn, atol=6e-3, rtol=6e-3)

# ===== 4. 显式测试 dht 路径 =====
print("\n>>> 单独测试 dht 路径")
# 只把 cuda_state 做损失，强制 dht 参与
cuda_state_grad = torch.randn_like(cuda_state)
loss_dht = (cuda_state * cuda_state_grad).sum()

# 重新求梯度（此时 dy=0，只有 dht 作用）
grad_dht = torch.autograd.grad(
    loss_dht, torch_inputs_cuda, retain_graph=False, allow_unused=True
)
# 期望：所有梯度非 None 且数值合理
for i, g in enumerate(grad_dht):
    assert g is not None, f"dht 路径 input[{i}] 梯度为 None"
    print(f"  input[{i}] 梯度范数={g.float().norm().item():.6f}")

print("✅ 反向传播测试通过！")
