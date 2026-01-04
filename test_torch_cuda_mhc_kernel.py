import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from keras import ops
# 指定显卡

torch.manual_seed(42)

# 导入 CUDA 实现
from rwkv_ops.mhc_kernel.torch_kernel.mhc_torch import (
    sinkhorn_knopp as cuda_sinkhorn,
    rmsnorm as cuda_rmsnorm,
    stream_mix as cuda_stream_mix,
    # stream_distribute as cuda_stream_distribute,
    stream_aggregate as cuda_stream_aggregate
)

# 2. 修改后的 Native 导入 (对应你提供的 native_keras_op 接口)
from rwkv_ops.mhc_kernel.native_keras_op import (
    sinkhorn_knopp as native_sinkhorn,
    rmsnorm as native_rmsnorm,
    stream_mix as native_stream_mix,
    stream_aggregate as native_stream_aggregate
)


def check_close(name, x1, x2, atol=1e-4, rtol=1e-4):
    """
    精度对比辅助函数
    """
    if x1 is None or x2 is None:
        print(f"❌ {name}: One of the gradients is None!")
        return False

    x1_val = x1.float().detach().cpu().numpy()
    x2_val = x2.float().detach().cpu().numpy()
    try:
        np.testing.assert_allclose(x1_val, x2_val, atol=atol, rtol=rtol)
        print(f"✅ {name} Pass")
        return True
    except AssertionError as e:
        # 计算最大绝对误差方便调试
        max_diff = np.max(np.abs(x1_val - x2_val))
        print(f"❌ {name} Fail (Max Diff: {max_diff:.6e})")

        print(e)
        error = (x1 - x2).abs()
        print(
            f"{float(torch.sum(error == 0)) / float(ops.cumprod(error.shape)[-1])}是完全一模一样的，平均误差是{error.mean()}"
        )
        return False


def rand_bfp(*shape):
    return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")


def make_grad(x):
    return x.detach().clone().requires_grad_(True)


# =====================================================
# 1. Sinkhorn Knopp 测试
# =====================================================
print("\n" + "=" * 20 + " Sinkhorn 测试 " + "=" * 20)
B, M, n = 8, 32, 4
s_inp_raw = rand_bfp(B, M, n, n).abs() + 0.1

# 前向测试
s_cuda_in = make_grad(s_inp_raw)
s_native_in = make_grad(s_inp_raw)

# 注意：为了公平比较，都转为 float32
cuda_s_out = cuda_sinkhorn(s_cuda_in.float(), num_iters=20, eps=1e-8)
native_s_out = native_sinkhorn(s_native_in.float(), num_iters=20, eps=1e-8)

check_close("Sinkhorn Forward", cuda_s_out, native_s_out)

# 反向测试
loss_cuda = (cuda_s_out**2).mean()
loss_native = (native_s_out**2).mean()

loss_cuda.backward()
loss_native.backward()

check_close("Sinkhorn Gradient", s_cuda_in.grad, s_native_in.grad, atol=1e-3)

# =====================================================
# 2. RMSNorm 测试
# =====================================================
print("\n" + "=" * 20 + " RMSNorm 测试 " + "=" * 20)
B, T, C = 4, 1024, 512
eps = 1e-5

x_raw = rand_bfp(B, T, C)


# 创建带梯度的副本
x_cuda = make_grad(x_raw)
x_native = make_grad(x_raw)

# 前向测试
# CUDA 版内部会自动处理转换，但为了对比我们显式调用
rms_cuda_out = cuda_rmsnorm(x_cuda, eps=eps)
rms_native_out = native_rmsnorm(x_native, eps=eps)

check_close(
    "RMSNorm Forward", rms_cuda_out, rms_native_out, atol=1e-2
)  # bf16 允许稍大误差

# 反向测试
# 构造一个稍微复杂的梯度信号
grad_output = torch.randn_like(rms_cuda_out) * 0.1

(rms_cuda_out * grad_output).sum().backward()
(rms_native_out * grad_output).sum().backward()

check_close("RMSNorm Input Grad (dx)", x_cuda.grad, x_native.grad, atol=1e-3)


# =====================================================
# 3. Stream Mix 测试 (n -> n)
# =====================================================
print("\n" + "=" * 20 + " Stream Mix 测试 " + "=" * 20)
n_stream = 4
M_mat = torch.randn(B, T, n_stream, n_stream, device="cuda").float()
inp_mix = rand_bfp(B, T, n_stream, C)

m_cuda, x_cuda = make_grad(M_mat), make_grad(inp_mix)
m_native, x_native = make_grad(M_mat), make_grad(inp_mix)

mix_cuda_out = cuda_stream_mix(x_cuda, m_cuda)
mix_native_out = native_stream_mix(x_native, m_native)

check_close("Mix Forward", mix_cuda_out, mix_native_out, atol=1e-3, rtol=1e-2)

(mix_cuda_out.float() ** 2).sum().backward()
(mix_native_out.float() ** 2).sum().backward()

check_close("Mix dx", x_cuda.grad, x_native.grad, atol=1e-3, rtol=1e-2)
check_close("Mix dM", m_cuda.grad, m_native.grad, atol=1e-3, rtol=5e-3)


# =====================================================
# 4. Stream Mix aggregate
# =====================================================
print("\n" + "=" * 20 + " Stream aggregate 测试 " + "=" * 20)
inp_mix = rand_bfp(B, T, n_stream, C)
H_pre = rand_bfp(B,T, n_stream)

H_pre_cuda, x_cuda = make_grad(H_pre), make_grad(inp_mix)
H_pre_native, x_native = make_grad(H_pre), make_grad(inp_mix)

native_out = native_stream_aggregate(x_native, H_pre_native)
cuda_out = cuda_stream_aggregate(x_cuda, H_pre_cuda)
check_close("Mix aggregate", cuda_out, native_out, atol=1e-3, rtol=1e-3)

(cuda_out.float() ** 2).sum().backward()
(native_out.float() ** 2).sum().backward()

check_close("Mix dx", x_cuda.grad, x_native.grad, atol=1e-3, rtol=1e-3)
check_close("Mix H_pre", H_pre_cuda.grad, H_pre_native.grad, atol=1e-3, rtol=1e-3)

print("\n" + "=" * 15 + " 所有 MHC 算子测试完成 " + "=" * 15)