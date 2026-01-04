[English Document](ENREADME.md)

# RWKV OPS 项目

> 由于 RWKV 将持续迭代，核心算子会随之更新。  
> 本仓专门维护「算子」本身，不维护 layer 与 model；尽可能提供各框架的 GPU 算子。  

### 当前支持
| 算子类型 | 框架支持 |
|----------|----------|
| GPU 算子 | PyTorch、JAX|
| 原生算子 | PyTorch、JAX、TensorFlow、NumPy |

> 未来若 Keras 生态扩展，可能支持 MLX、OpenVINO。  
> 注意：本库依赖 `keras`。

---

## 安装

```bash
pip install rwkv_ops
```

当然pip包对于编译的算子pip uninstal没法删干净，所有可以试着从源码安装
```bash
git clone https://github.com/pass-lin/rwkv_ops.git
cd rwkv_ops
bash install.sh
```
---

## 环境变量

| 变量名 | 含义 | 取值 | 默认值 | 优先级 |
|---|---|---|---|---|
| `KERAS_BACKEND` | Keras 后端 | `jax` / `torch` / `tensorflow` / `numpy` | — | 低 |
| `KERNEL_BACKEND` | 算子后端 | `jax` / `torch` / `tensorflow` / `numpy` | `torch` | **高** |
| `KERNEL_TYPE` | 实现类型 | `triton` / `cuda` / `native` | `cuda` | — |

> 若 `KERNEL_BACKEND` 有值，直接采用；若为空，则用 `KERAS_BACKEND`；两者皆空则默认 `torch`。  

---


## [MHC 算子](https://arxiv.org/abs/2512.24880)

虽然和RWKV无关，但是我懒得再开一个包做分发了，就集成在这了吧。 

### 背景

在多头架构中，传统做法通常是简单的线性变换或加权。MHC 引入了 **Sinkhorn-Knopp** 算法，将控制权重约束在双稳态矩阵空间内，从而保证了信息流动的守恒性和稳定性。由于这些操作涉及大量的中间变量和迭代计算，原生实现的显存占用极高。本库提供的 CUDA 算子通过 **算子融合（Operator Fusion）** 技术，显著降低了显存消耗并提升了运行速度。

需要注意的是，这个仓库只提供最朴素的cuda实现。使用了https://github.com/AndreSlavescu/mHC.cu 和 Gemini协助完成。现在的代码可以成功通过测试

---

### MHC 算子列表

| 算子名称 | 核心功能 |
| --- | --- |
| `mhc_pre_op` | **前置融合算子**：处理输入流聚合与 Sinkhorn 矩阵准备。 |
| `mhc_post_op` | **后置融合算子**：处理层输出分发与多头残差融合。 |
| `sinkhorn_knopp` | 矩阵双稳态归一化（支持高精度反向迭代）。 |
| `rmsnorm` | 针对 MHC 输入分布优化的 RMS 归一化。 |
| `stream_aggregate` | 特征流加权聚合（多流转单流）。 |
| `stream_distribute` | 特征流加权分发（单流转多流）。 |
| `stream_mix` | 特征流间的动态混合（Cross-head Mixing）。 |

---

### MHC 典型集成流程

MHC 算子的使用流程，这是一个取代resnet的框架。我懒得写容器了，大概的使用流程如下所示：

```python
from rwkv_ops import rmsnorm, mhc_pre_op, mhc_post_op

# 1. 归一化输入
x_norm = rmsnorm(x_expanded)

# 2. 生成原始控制参数 (通常通过 Linear 层)
# h_res_raw: [B, T, N, N], h_pre_raw/h_post_raw: [B, T, N]
h_res_raw, h_pre_raw, h_post_raw = linear_and_reshape(x_norm)

# 3. MHC 前置处理 (Fused Kernel)
x_layer_in, H_post, H_res = mhc_pre_op(
    x_expanded, h_pre_raw, h_post_raw, h_res_raw, num_iters=20
)

# 4. 执行核心层逻辑 (x_layer_in 为聚合后的单流 [B, T, C])
layer_out = YourCoreLayer(x_layer_in)

# 5. MHC 后置处理 (Fused Kernel)
x_next = mhc_post_op(layer_out, x_expanded, H_post, H_res)

```

---

### 算子详细定义

#### 1. `mhc_pre_op` (Fused Pre-computation)

**融合说明**：该算子等价于以下原生操作的融合：

* 对 `h_pre_raw` 执行 `Sigmoid` 得到前置门控。
* 对 `h_res_raw` 执行 `Exp` + `Sinkhorn-Knopp` 得到双稳态权重矩阵。
* 对 `x_expanded` 执行 `stream_aggregate`（按头聚合）。
* **融合优势**：避免了存储巨大的 Exp 矩阵和中间迭代状态，显存占用降低约 80%。

**接口定义**：

* **输入**:
* `x_expanded` : 展开后的  个特征头。
* `h_pre_raw` : 原始前置系数。
* `h_post_raw` : 原始后置系数。
* `h_res_raw` : 原始 Sinkhorn 输入。


* **返回**:
* `x_layer_in` : 融合后的层输入。
* `H_pre`, `H_post`, `H_res`: 供 `post_op` 及反向传播使用的归一化系数。



#### 2. `mhc_post_op` (Fused Post-computation)

**融合说明**：该算子等价于以下原生操作的融合：

* 执行 `stream_distribute` 将单流输出映射回多流。
* 执行 `stream_mix` 利用 `H_res` 进行头间信息交换。
* 对 `H_post` 执行  并作为残差门控。
* 执行多头残差加法：。

**接口定义**：

* **输入**: `layer_out` , `x_expanded` , `H_post`, `H_res`。
* **返回**: `x_next` 。


#### 3. `sinkhorn_knopp`

* **定义**: 。
* **特点**: CUDA 实现采用双向迭代，其梯度计算通过求解伴随状态方程实现，比直接对迭代过程进行自动微分更稳定且更省显存。

#### 4. `rmsnorm`

* **定义**: 标准的归一化与流操作，但在 CUDA 实现中针对 MHC 的特征分布（通常在  维度上）进行了特定的访存优化。参考mHC的实现，这个rmsnorm是不带参数的。


### 5. `stream_aggregate`   

**功能定义**：
该算子执行加权空间压缩。它将  个独立的特征头（Streams）根据给定的权重向量进行线性加权求和，坍缩为一个统一的特征表示。

* **数学表达式**：，其中 ，。
* **等价操作**：相当于执行了 `torch.einsum('btn,btnc->btc', weights, x)`。

**接口定义**：

* **输入**:
* `x`:  多流特征张量。
* `weights`:  每个流对应的权重系数。


* **返回**:
* `out`:  聚合后的单流特征。



---

### 6. `stream_distribute`  

**功能定义**：
该算子执行特征的空间广播与重加权。它将一个单流特征复制到  个通道，并分别乘以对应的分发权重。

* **数学表达式**：，其中 。
* **等价操作**：相当于执行了 `x.unsqueeze(2) * weights.unsqueeze(-1)`。

**接口定义**：

* **输入**:
* `x`:  待分发的单流特征（通常来自核心层输出）。
* `weights`:  分发到每个头的权重系数。


* **返回**:
* `out`:  分发后的多流特征张量。



---

### 7. `stream_mix` (流混合)

**功能定义**：
该算子是 MHC 实现头间通信（Cross-head Communication）的关键。它利用一个  的变换矩阵（通常是 Sinkhorn 归一化后的矩阵），在不同的特征头之间进行线性重组。

* **数学表达式**：。
* **等价操作**：相当于执行了 `torch.einsum('btnm,btmc->btnc', gate, x)`。
* **物理意义**：实现了信息的“非局部”重分配，使得每个头都能吸收来自其他头的信息。

**接口定义**：

* **输入**:
* `x`:  原始多流特征。
* `gate`:  混合矩阵（控制信息流动的开关与强度）。


* **返回**:
* `out`:  混合后的多流特征。



---

### MHC 实现状态

| Framework | cuda | triton | native |
| --- | --- | --- | --- |
| **PyTorch** | ✅ | ❌ | ✅ |
| **JAX** | ❌ | ❌ | ✅ |
| **TensorFlow** | ❌ | ❌ | ✅ |
| **NumPy** | ❌ | ❌ | ✅ |


---



```python
from rwkv_ops import generalized_delta_rule,generalized_delta_rule_inference  # 或 from rwkv_ops import rwkv7_op，完全等价
#generalized_delta_rule_inference的入口和这个接口一致
#但是generalized_delta_rule_inference是没有梯度只支持inference的
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
generalized_delta_rule_inference和generalized_delta_rule的区别是前者没有梯度。因为不需要存储激活值，所以可以节省一部分显存。

### cuda-kernel 特殊用法

- torch-cuda和jax-cuda kernel 下 `head_size` 也是一个 kernel 参数，默认为 64。  
- 若 `head_size ≠ 64`，请使用：

```python
from rwkv_ops import get_generalized_delta_rule

rwkv7_op, rwkv7_op_inference, USE_TRITON_KERNEL = get_generalized_delta_rule(
    your_head_size, KERNEL_TYPE="cuda"
)
```

- `USE_TRITON_KERNEL` 为常量，标记是否使用 chunkwise 算子。  
- 两者 padding 处理逻辑不同：

```python
if padding_mask is not None:
    w += (1 - padding_mask) * -1e9
```
- 对于上面的代码，基于循环的算子可以针对left pading和right pading都能成功处理。
- 而如果用的是chunkwise算子，建议统一left padding，如果是cuda或者原生，则都left right都能正确处理


### rwkv7op 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ✅     | ✅     |
| JAX         | ✅   | ✅     | ✅     |
| TensorFlow  | ⚠️    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| MLX       | ⚠️   | ❌     | ❌     |

---
1. `native` 为原生算子，无 chunkwise，速度慢且显存高。
2. `triton` 使用的是chunkwise算法实现，速度快，并行度高，缺点是精度很差，介意勿用
3. `cuda` 为基于 CUDA 的原生算子，速度很快，并且kernel内部使用fp32实现，所以精度也很高。缺点就是长序列的时候比较吃亏跑不满。
4. tensorflow的CUDA实现只支持前向计算，是没有梯度的。并且这个是使用jax的cuda实现实现的，你需要保证你能够成功运行jax的cuda kernel。
5. tensorflow kernel只支持eager
6. 因为MLX还没合并到keras，所以原生算子暂不支持。但是我们提供了一个前向的算子。
## rwkv7_op_rnn 使用方法

### 背景
这是RWKV7 OP的特殊情况，就是我们只考虑长度=1的情况。专门用于推理的decode阶段的加速

### 使用方法

```python
from rwkv_ops import rwkv7_op_rnn
def rwkv7_op_rnn(
        r: jnp.ndarray,
        w: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        initial_state: Optional[jnp.ndarray] = None,
        output_final_state: bool = True,
        head_first: bool = False,
    )
            """
        单步广义 delta 规则（仅前向）
        参数:
            r,w,k,v,a,b: 输入张量，形状必须为 (B, 1, H, K) 或 (B, H, 1, K)
            initial_state: 可选 (B, H, K, K) 初始状态，None 则零初始化
            output_final_state: 是否同时返回最后状态
            head_first: 是否将 head 维提前
        返回:
            out: (B, 1, H, K)  与输入 dtype 一致
            last_state: (B, H, K, K) 当 output_final_state=True
        """
```
### rwkv7_op_rnn 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ❌     | ✅     |
| JAX         | ✅   | ❌     | ✅     |
| TensorFlow  | ⚠️    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |

1. tf的cuda实现依赖于jax的cuda实现，所以需要安装jax
2. native实现我们直接复用了rwkv7_op的native实现
3. **这个算子没有梯度**
4.  tensorflow kernel只支持eager

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
