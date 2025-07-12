import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_PRINT_AUTOTUNING"] = "-1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_LOG_COMPILE"] = "False"
import logging

# 设置日志级别为ERROR，屏蔽INFO和WARNING
logging.basicConfig(level=logging.ERROR)
import numpy as np
from keras import ops
import torch


def test_output(output_jax, output_torch):
    tol = 5e-3
    for i in range(len(output_torch)):
        if output_jax[i] == None or output_torch[i] == None:
            continue
        out_jax = ops.convert_to_numpy(ops.cast(output_jax[i], "float32"))
        out_torch = output_torch[i].float().cpu().numpy()
        flag = np.allclose(out_jax, out_torch, rtol=max(tol, 1e-5), atol=tol)
        if np.sum(out_jax - out_torch) == 0:
            print(f"第{i + 1}个输出结果完全一致")
        else:
            print(f"第{i + 1}个输出函数的校验结果是:{flag}")
        if np.sum(np.isnan(out_jax)):
            print("存在NAN值")
        else:
            print("不存在NAN值")


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


T = 128
B = 2
H = 6
K = 128
np.random.seed(0)
inputs = [np.random.randn(B, T, H, K) for _ in range(30)]
dht = np.random.randn(B, T, H, T)
d0 = np.random.randn(B, H, T, T)
h = np.random.randn(B, 8, H, T, T)

from rwkv_ops.rwkv7_kernel.torch_op import chunk_dplr_bwd as torch_chunk_dplr_bwd


jax_inputs = [ops.convert_to_tensor(t, dtype="bfloat16") for t in inputs]
torch_inputs = [torch.from_numpy(t).bfloat16().cuda() for t in inputs]


from rwkv_ops.rwkv7_kernel.torch_op import chunk_dplr_fwd as torch_chunk_dplr_fwd


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


a = np.array(-normalize(jax_inputs[3], dim=-1, p=2.0), "float32")
b = np.array(normalize(jax_inputs[3], dim=-1, p=2.0), "float32")
gk = np.array(-ops.exp(-ops.softplus(jax_inputs[5])), "float32")
from rwkv_ops.rwkv7_kernel.jax_op import chunk_dplr_fwd as chunk_dplr
from rwkv_ops.rwkv7_kernel.jax_op import chunk_dplr_bwd, CHUNKSIZE

output_torch = torch_chunk_dplr_bwd(
    q=torch_inputs[0],
    k=torch_inputs[1],
    v=torch_inputs[2],
    a=torch.from_numpy(a).bfloat16().cuda(),
    b=torch.from_numpy(b).bfloat16().cuda(),
    gk=torch.from_numpy(gk).bfloat16().cuda(),
    scale=1,
    do=torch.from_numpy(d0).to(torch_inputs[0]),
    dht=torch.from_numpy(dht).to(torch_inputs[0]),
    BT=CHUNKSIZE,
    initial_state=None,
)

output_jax = chunk_dplr_bwd(
    q=jax_inputs[0],
    k=jax_inputs[1],
    v=jax_inputs[2],
    a=ops.convert_to_tensor(a, jax_inputs[2].dtype),
    b=ops.convert_to_tensor(b, jax_inputs[2].dtype),
    gk=ops.convert_to_tensor(gk, jax_inputs[2].dtype),
    scale=1,
    do=ops.convert_to_tensor(d0, dtype=jax_inputs[0].dtype),
    dht=ops.convert_to_tensor(dht, dtype=jax_inputs[0].dtype),
    initial_state=None,
)

output_torch = [t for t in output_torch if t is not None]
print("校验chunk_dplr_fwd_bwd函数")
test_output(output_jax, output_torch)


output_jax = chunk_dplr(
    q=jax_inputs[0],
    k=jax_inputs[1],
    v=jax_inputs[2],
    a=ops.convert_to_tensor(a, jax_inputs[2].dtype),
    b=ops.convert_to_tensor(b, jax_inputs[2].dtype),
    gk=ops.convert_to_tensor(gk, jax_inputs[2].dtype),
    scale=1,
    initial_state=None,
    output_final_state=True,
    chunk_size=CHUNKSIZE,
)

output_torch = torch_chunk_dplr_fwd(
    q=torch_inputs[0],
    k=torch_inputs[1],
    v=torch_inputs[2],
    a=torch.from_numpy(a).bfloat16().cuda(),
    b=torch.from_numpy(b).bfloat16().cuda(),
    gk=torch.from_numpy(gk).bfloat16().cuda(),
    scale=1,
    initial_state=None,
    output_final_state=True,
)

print("校验chunk_dplr_fwd_fwd函数")
test_output(output_jax, output_torch)

from rwkv_ops.rwkv7_kernel.jax_op import generalized_delta_rule

jax_chunkout, jax_state = generalized_delta_rule(
    r=jax_inputs[0],
    k=jax_inputs[1],
    v=jax_inputs[2],
    a=ops.convert_to_tensor(a, jax_inputs[2].dtype),
    b=ops.convert_to_tensor(b, jax_inputs[2].dtype),
    w=ops.convert_to_tensor(gk, jax_inputs[2].dtype),
)


try:
    from rwkvfla.ops.rwkv7 import chunk_rwkv7
except:
    from rwkvfla.ops.rwkv7 import chunk_rwkv7 as chunk_rwkv7_fla

from rwkv_ops.rwkv7_kernel.torch_op import (
    generalized_delta_rule as generalized_delta_rule_torch,
)

my_chunkout, my_state = generalized_delta_rule_torch(
    r=torch_inputs[0],
    k=torch_inputs[1],
    v=torch_inputs[2],
    a=torch.from_numpy(a).bfloat16().cuda(),
    b=torch.from_numpy(b).bfloat16().cuda(),
    w=torch.from_numpy(gk).bfloat16().cuda(),
    output_final_state=True,
)
print("校验jax和torch实现kernel前向的精度,主要误差来自jnp.exp和torch.exp")
torch_out_numpy = my_chunkout.cpu().float().numpy()
jax_out_numpy = np.array(ops.cast(jax_chunkout, "float32"))

print(
    "最大正数差异%f,平均差异%f,%f的输出完全一致"
    % (
        np.max(np.abs(torch_out_numpy - jax_out_numpy)),
        np.mean(np.abs(torch_out_numpy - jax_out_numpy)),
        np.sum(torch_out_numpy - jax_out_numpy == 0)
        / np.cumprod(torch_out_numpy.shape)[-1],
    )
)
fla_chunkout, fla_state = chunk_rwkv7_fla(
    r=torch_inputs[0],
    k=torch_inputs[1],
    v=torch_inputs[2],
    a=torch.from_numpy(a).bfloat16().cuda(),
    b=torch.from_numpy(b).bfloat16().cuda(),
    w=torch.from_numpy(gk).bfloat16().cuda(),
    scale=1,
    initial_state=None,
    output_final_state=True,
)
print("校验fla和我的实现前向的精度")
print(
    "最大正数差异%f,平均差异%f"
    % (
        (fla_chunkout - my_chunkout).abs().max(),
        (fla_chunkout - my_chunkout).abs().mean(),
    )
)


mask = ops.concatenate([ops.ones([B, T]), ops.zeros([B, T])], axis=1)
mask = ops.cast(mask, jax_chunkout.dtype)[:, :, None, None]


def padding_input(x):
    return ops.concatenate([x, x], axis=1)


w = padding_input(ops.convert_to_tensor(gk, jax_inputs[2].dtype))
w += (1 - mask) * -1e9
jax_pad_chunkout, jax_pad_state = generalized_delta_rule(
    r=padding_input(jax_inputs[0]) * mask,
    k=padding_input(jax_inputs[1]) * mask,
    v=padding_input(jax_inputs[2]) * mask,
    a=padding_input(ops.convert_to_tensor(a, jax_inputs[2].dtype)) * mask,
    b=padding_input(ops.convert_to_tensor(b, jax_inputs[2].dtype)) * mask,
    w=w,
)

print("padding 后state的输出完全一致:%s" % str(ops.sum(jax_pad_state - jax_state) == 0))


# 定义 loss 函数
def loss_fn(output):
    return output.sum()


print("验证 chunk_rwkv7_fla 和 chunk_rwkv7 的反向传播精度：")
# 设置 requires_grad=True
for t in torch_inputs:
    t.requires_grad = True

# 其他张量也需要设置 requires_grad=True
a_tensor = torch.from_numpy(a).bfloat16().cuda()
a_tensor.requires_grad = True

b_tensor = torch.from_numpy(b).bfloat16().cuda()
b_tensor.requires_grad = True

gk_tensor = torch.from_numpy(gk).bfloat16().cuda()
gk_tensor.requires_grad = True
from copy import deepcopy

fla_input = deepcopy(
    dict(
        r=torch_inputs[0],
        k=torch_inputs[1],
        v=torch_inputs[2],
        a=a_tensor,
        b=b_tensor,
        w=gk_tensor,
    )
)
outputs1 = chunk_rwkv7_fla(
    **fla_input,
    scale=1,
    initial_state=None,
    output_final_state=True,
)

my_input = deepcopy(
    dict(
        r=torch_inputs[0],
        k=torch_inputs[1],
        v=torch_inputs[2],
        a=a_tensor,
        b=b_tensor,
        w=gk_tensor,
    )
)

outputs2 = chunk_rwkv7_fla(
    **my_input,
    scale=1,
    initial_state=None,
    output_final_state=True,
)


loss1 = loss_fn(outputs1[0])
loss2 = loss_fn(outputs2[0])

# 反向传播
loss1.backward(retain_graph=True)
loss2.backward(retain_graph=True)
my_input = list(my_input.items())
fla_input = list(fla_input.items())

for i in range(len(my_input)):
    print(
        "%s变量梯度的最大正数差异%f,平均差异%f"
        % (
            my_input[i][0],
            torch.max(torch.abs(my_input[i][1].grad - fla_input[i][1].grad)),
            torch.mean(torch.abs(my_input[i][1].grad - fla_input[i][1].grad)),
        )
    )
