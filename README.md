[English Document](ENREADME.md)

# RWKV OPS 项目

> 由于 RWKV 将持续迭代，核心算子会随之更新。  
> 本仓专门维护「算子」本身，不维护 layer 与 model；尽可能提供各框架的 GPU 算子。  

### 当前支持
| 算子类型 | 框架支持 |
|----------|----------|
| GPU 算子 | PyTorch、JAX（TensorFlow 待 Google 支持 Triton 后上线） |
| 原生算子 | PyTorch、JAX、TensorFlow、NumPy |

> 未来若 Keras 生态扩展，可能支持 MLX、OpenVINO。  
> 注意：本库依赖 `keras`。

---

## 安装

```bash
pip install rwkv_ops
```

---

## 环境变量

| 变量名 | 含义 | 取值 | 默认值 | 优先级 |
|---|---|---|---|---|
| `KERAS_BACKEND` | Keras 后端 | `jax` / `torch` / `tensorflow` / `numpy` | — | 低 |
| `KERNEL_BACKEND` | 算子后端 | `jax` / `torch` / `tensorflow` / `numpy` | `torch` | **高** |
| `KERNEL_TYPE` | 实现类型 | `triton` / `cuda` / `native` | — | — |

> 若 `KERNEL_BACKEND` 有值，直接采用；若为空，则用 `KERAS_BACKEND`；两者皆空则默认 `torch`。  
> `native` 为原生算子，无 chunkwise，速度慢且显存高。

---

## rwkv7op 使用方法

```python
from rwkv_ops import generalized_delta_rule  # 或 from rwkv_ops import rwkv7_op，完全等价

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
    分块 Delta Rule 注意力接口。

    Args:
        q:  [B, T, H, K]
        k:  [B, T, H, K]
        v:  [B, T, H, V]
        a:  [B, T, H, K]
        b:  [B, T, H, K]
        gk: [B, T, H, K]  # decay term in log space!
        initial_state: 初始状态 [N, H, K, V]，N 为序列数
        output_final_state: 是否返回最终状态
        head_first: 是否 head-first 格式，不支持变长

    Returns:
        o:           输出 [B, T, H, V] 或 [B, H, T, V]
        final_state: 最终状态 [N, H, K, V] 或 None
    """
```

### torch-cuda 特殊用法

- torch-cuda 下 `head_size` 也是一个 kernel 参数，默认为 64。  
- 若 `head_size ≠ 64`，请使用：

```python
from rwkv_ops import get_generalized_delta_rule

generalized_delta_rule, RWKV7_USE_KERNEL = get_generalized_delta_rule(
    your_head_size, KERNEL_TYPE="cuda"
)
```

- `RWKV7_USE_KERNEL` 为常量，标记是否使用 chunkwise 算子。  
- 两者 padding 处理逻辑不同：

```python
if padding_mask is not None:
    if RWKV7_USE_KERNEL:
        w += (1 - padding_mask) * -1e9
    else:
        w = w * padding_mask + 1 - padding_mask
```

---

### rwkv7op 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ✅     | ✅     |
| JAX         | ❌   | ✅     | ✅     |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |

---

## rwkv6op 使用方法

### PyTorch 使用注意事项

- 安装依赖：`keras`、`ninja`、完整的 CUDA 工具包。
- 若使用 VS Code + 虚拟环境调试，请务必在终端手动激活虚拟环境，再运行代码，否则 ninja 可能无法工作。
- 虽然 PyTorch 在「虚拟环境中的 CUDA 版本」与「全局 CUDA 版本」不一致时仍可正常运行，但强烈建议保持一致。
- PyTorch 限制：同一程序内只能实例化 **一个** `RWKV6_OP` 对象；算子线程安全（无状态），可在多处调用。

### JAX 使用注意事项

- 安装依赖：`keras`、`gcc`、`pybind11`、完整的 CUDA 工具包。
- 即使通过虚拟环境为 JAX 安装 CUDA，也必须在系统级安装完整 CUDA；两者版本需一致，以保证 JAX 并行编译速度。
- JAX 编译依赖 `/usr/local/cuda` 软链接，如不存在请手动创建：
  ```shell
  sudo ln -sf /usr/local/cuda-12.4 /usr/local/cuda
  ```
- 确保 `nvcc -V` 正常输出，且 `which nvcc` 指向正确版本。
- JAX 限制：同一程序内只能实例化 **一个** `RWKV6_OP` 对象；算子线程安全（无状态），可在多处调用。
- JAX ≥ 0.6.0 不再使用 CUDA 算子，默认使用原生算子；推荐 0.4.34。

### TensorFlow 使用注意事项

- 仅提供基于原生 API 的 `RWKV6` 算子，仅用于推理，效率较低。

---

### 使用方法
需要注意的是，和rwkv7写成函数的形式不一样，RWKV6的op是一个类，需要实例化。
```python
from rwkv_ops import RWKV6_OP

operator = RWKV6_OP(
    head_size=64,               # 头大小，不确定时填 64
    max_sequence_length=4096,   # 训练最大序列长度；推理不受限
    ops_loop=False              # 可选：序列长度=1 时是否用上层 API 替代 CUDA
)
```

#### 调用

```python
y, y_state = operator(
    r, k, v, w, u,
    with_state=False,   # 是否使用自定义初始状态 / 输出结束状态
    init_state=None,    # 初始状态 [n_state, num_heads, head_size, head_size]
    state_map=None      # int32 一维数组，长度=batch_size，定义 init_state 映射
)
```

| 参数 | 形状 | 说明 |
|---|---|---|
| r, k, v, w | (batch_size, seq_len, hidden_size) | — |
| u | (num_heads, head_size) 或 (hidden_size,) | — |
| init_state | (n_state, num_heads, head_size, head_size) | n_state=1 时所有样本共用；n_state=batch_size 时一一对应 |
| state_map | (batch_size,) | 指定每个样本用到的 init_state 索引 |

| 返回值 | 形状 | 说明 |
|---|---|---|
| y | (batch_size, seq_len, hidden_size) | 输出 |
| y_state | (batch_size, num_heads, head_size, head_size) 或 None | 结束状态 |

---

### 分布式小贴士

- 算子本身无分布式支持；PyTorch 可直接用多线程分布式。
- JAX 需通过 `shard_map` 包装（示例）：

```python
import os
os.environ['KERAS_BACKEND'] = 'jax'

import jax, jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from functools import partial
from rwkv_ops import RWKV6_OP

batch_size, seq_length = 24, 512
head_size, num_heads = 64, 32
hidden_size = head_size * num_heads

mesh = Mesh(jax.devices('gpu'), axis_names=('device_axis',))
device_ns = NamedSharding(mesh, P('device_axis'))

operator = RWKV6_OP(head_size=head_size, max_sequence_length=seq_length)

@partial(shard_map,
         mesh=mesh,
         in_specs=(P('device_axis'),) * 5,
         out_specs=(P('device_axis'), P('device_axis')),
         check_rep=False)
def call_kernel(r, k, v, w, u):
    # 去掉最外 device 维度
    r, k, v, w, u = map(jnp.squeeze, (r, k, v, w, u))
    y, ys = operator(r, k, v, w, u, with_state=True)
    return jnp.expand_dims(y, 0), jnp.expand_dims(ys, 0)

# 构造输入并放置到对应设备
keys = jax.random.split(jax.random.PRNGKey(0), 5)
inputs = [jax.random.normal(k, (mesh.size, batch_size, seq_length, hidden_size)) for k in keys]
inputs_r, inputs_k, inputs_v, inputs_w, inputs_u = map(
    lambda x: jax.device_put(x, device_ns), inputs)
inputs_u = inputs_u[:, :, 0]  # (devices, hidden_size)

# 可选：jax.jit(call_kernel, ...) 加速
outputs_y, y_state = call_kernel(inputs_r, inputs_k, inputs_v, inputs_w, inputs_u)

print(outputs_y.shape, outputs_y.sharding)
print(y_state.shape, y_state.sharding)
```

---

### rwkv6op 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ❌     | ✅     |
| JAX         | ⚠️   | ❌     | ✅     |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |

⚠️ JAX 的 CUDA 实现仅适用于 < 0.6.0，推荐 0.4.34。
