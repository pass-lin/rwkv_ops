[English Document](ENREADME.md)

# RWKV OPS 项目

> 由于 RWKV 将持续迭代，核心算子会随之更新。  
> 本仓专门维护「算子」本身，不维护 layer 与 model；尽可能提供各框架的 GPU 算子。  

<a id="当前支持"></a>
### 当前支持
| 算子类型 | 框架支持 |
|----------|----------|
| GPU 算子 | PyTorch、JAX|
| 原生算子 | PyTorch、JAX、TensorFlow、NumPy |

> 未来若 Keras 生态扩展，可能支持 MLX、OpenVINO。  

> 注意：本库依赖 `keras`。

## 目录

  - [当前支持](#当前支持)
- [安装](#安装)
- [环境变量](#环境变量)
- [mhcop 使用方法](#mhcop-使用方法)
  - [快速开始](#快速开始)
  - [函数接口说明](#函数接口说明)
    - [`mhc_pre_op`](#mhc_pre_op)
    - [`mhc_post_op`](#mhc_post_op)
  - [mhcop 实现状态](#mhcop-实现状态)
- [rwkv7op 使用方法](#rwkv7op-使用方法)
  - [cuda-kernel 特殊用法](#cuda-kernel-特殊用法)
  - [rwkv7op 实现状态](#rwkv7op-实现状态)
- [rwkv7_op_rnn 使用方法](#rwkv7_op_rnn-使用方法)
  - [背景](#背景)
  - [使用方法](#使用方法)
  - [rwkv7_op_rnn 实现状态](#rwkv7_op_rnn-实现状态)
- [rwkv7op_sane 使用方法](#rwkv7op_sane-使用方法)
  - [rwkv7op_sane 实现状态](#rwkv7op_sane-实现状态)
- [rwkv7_op_sane_rnn 使用方法](#rwkv7_op_sane_rnn-使用方法)
  - [rwkv7_op_sane_rnn 实现状态](#rwkv7_op_sane_rnn-实现状态)
- [gdn_recurrent 使用方法](#gdn_recurrent-使用方法)
  - [函数接口说明](#函数接口说明-1)
    - [`gated_delta_net_recurrent`](#gated_delta_net_recurrent)
    - [`gated_delta_net_recurrent_inference`](#gated_delta_net_recurrent_inference)
    - [`gated_delta_net_recurrent_single_step`](#gated_delta_net_recurrent_single_step)
  - [gdn_recurrent 实现状态](#gdn_recurrent-实现状态)
- [gdn_recurrent_sane 使用方法](#gdn_recurrent_sane-使用方法)
  - [函数接口说明](#函数接口说明-2)
    - [`gated_delta_net_recurrent_sane`](#gated_delta_net_recurrent_sane)
    - [`gated_delta_net_recurrent_sane_inference`](#gated_delta_net_recurrent_sane_inference)
    - [`gated_delta_net_recurrent_sane_single_step`](#gated_delta_net_recurrent_sane_single_step)
  - [gdn_recurrent_sane 实现状态](#gdn_recurrent_sane-实现状态)
- [gdn_chunk 使用方法](#gdn_chunk-使用方法)
  - [函数接口说明](#函数接口说明-3)
    - [`gated_delta_net_chunk`](#gated_delta_net_chunk)
  - [gdn_chunk 实现状态](#gdn_chunk-实现状态)
- [gdn_chunk_sane 使用方法](#gdn_chunk_sane-使用方法)
  - [函数接口说明](#函数接口说明-4)
    - [`gated_delta_net_chunk_sane`](#gated_delta_net_chunk_sane)
  - [gdn_chunk_sane 实现状态](#gdn_chunk_sane-实现状态)
- [rwkv6op 使用方法](#rwkv6op-使用方法)
- [分布式并行（JAX）](#分布式并行)
  - [PyTorch 使用注意事项](#pytorch-使用注意事项)
  - [JAX 使用注意事项](#jax-使用注意事项)
  - [TensorFlow 使用注意事项](#tensorflow-使用注意事项)
  - [使用方法](#使用方法-1)
  - [rwkv6op 实现状态](#rwkv6op-实现状态)
- [测试](#测试)

---

<a id="安装"></a>
## 安装

```bash
pip install rwkv_ops
```


<a id="环境变量"></a>
## 环境变量

| 变量名 | 含义 | 取值 | 默认值 | 优先级 |
|---|---|---|---|---|
| `KERAS_BACKEND` | Keras 后端 | `jax` / `torch` / `tensorflow` / `numpy` / `openvino` | — | 低 |
| `KERNEL_BACKEND` | 算子后端 | `jax` / `torch` / `tensorflow` / `numpy` / `openvino` | `torch` | **高** |
| `KERNEL_TYPE` | 实现类型 | `triton` / `cuda` / `native` | `cuda` | — |
| `RWKV_OPS_PALLAS_AUTOTUNE` | Pallas autotune 开关 | `1` / `0` | `0` | — |
| `RWKV_OPS_KERAS_NATIVE` | 强制 native 为纯 keras ops | `1` / `0` | `0` | — |

> `KERNEL_TYPE=native` 时按后端与平台分发实现：jax + GPU/TPU 用 Pallas kernel（rwkv7/rwkv7_sane）；torch + 非 CPU 用 Triton kernel（rwkv7/rwkv7_sane/mhc，pip 版 torch 自带 triton）；其余（CPU、mhc 的 jax 侧等）为纯 Keras ops。设 `RWKV_OPS_KERAS_NATIVE=1` 可强制全部为纯 Keras ops（调试用）。

> 若 `KERNEL_BACKEND` 有值，直接采用；若为空，则用 `KERAS_BACKEND`；两者皆空则默认 `torch`。  

---
<a id="mhcop-使用方法"></a>
## mhcop 使用方法

[mHC (Multi-Head Control)](https://arxiv.org/pdf/2512.24880) 是 DeepSeek 实现的一种取代 ResNet 的新残差交互机制。它将传统的单流残差扩展为多流并行，并引入动态聚合与分发。

本仓提供了基于 **Triton** 实现的 Keras 算子。由于这是作者的一个 Triton 练手项目，性能优化尚未达到极限：
* **JAX 端**：XLA 的融合能力极其恐怖，导致 Triton 的提速并不明显（Native 耗时约 ResNet 的 1.5x，Triton 约 1.27x，DeepSeek 原版约 1.06x）。但在 **显存** 方面，Triton 算子通过手写 VJP 强制重计算，在 `128x1024x4x768` 规模下可比 JAX Native 节省 **3~4GB** 显存。注意这里说的是模型整体。  
* **Torch 端**：由于 `torch.compile` 对此类复杂逻辑的融合效率远不如 XLA，Triton 算子表现出巨大的优势（Pre-Op 提速可达 8 倍，Post-Op 约 3 倍）。**建议 Torch 用户默认开启。** 注意这里说的是单算子，我懒得测torch的模型整体情况了。  

<a id="快速开始"></a>
### 快速开始

```python
from rwkv_ops import mhc_pre_op, mhc_post_op

# 也可以显式获取指定后端（默认为 triton）
# mhc_pre_op, mhc_post_op = get_mhc_kernel("triton")

# 在每一层核心逻辑（Attention/FFN）前后的调用示例：
# 1. 预处理：多流聚合为单流
x_layer_in, h_post, h_res = mhc_pre_op(
    x, alpha_pre, alpha_post, alpha_res, phi, 
    bias_pre, bias_post, bias_res, n=4
)

# 2. 核心层计算
x_layer_out = attention(x_layer_in)

# 3. 后处理：分发回多流并进行流混合
x_next = mhc_post_op(x_layer_out, x, h_post, h_res)
```

---

<a id="函数接口说明"></a>
### 函数接口说明

<a id="mhc_pre_op"></a>
#### `mhc_pre_op`
将多流特征聚合为核心层输入，并生成后续所需的投影系数。

| 参数 | 形状 | 说明 |
|---|---|---|
| x | (B, T, n, C) | 多流输入特征 |
| alpha_pre/post/res | (1,) | 聚合、分发、残差三个分支的缩放标量系数 |
| phi | (n*C, n*(n+2)) | 动态投影矩阵 |
| bias_pre/post/res | (M,) | 各分支对应的偏置项 |
| n | int | 扩展流的数量（Head 数量） |
| num_iters | int | Sinkhorn-Knopp 迭代次数（默认 20） |

| 返回值 | 形状 | 说明 |
|---|---|---|
| x_layer_in | (B, T, C) | 聚合后的单流特征，喂给 Attention/FFN |
| h_post_raw | (B, T, n) | 分发权重（未激活），用于 Post-Op |
| H_res | (B, T, n, n) | 双随机残差混合矩阵，用于 Post-Op |

---

<a id="mhc_post_op"></a>
#### `mhc_post_op`
将核心层输出通过门控权重分发回多流，并利用混合矩阵更新流状态。

| 参数 | 形状 | 说明 |
|---|---|---|
| layer_out | (B, T, C) | 核心层（Attention/FFN）的输出 |
| x_expanded | (B, T, n, C) | Pre-Op 之前的多流状态（残差路径数据） |
| h_post_raw | (B, T, n) | 来自 Pre-Op 的分发权重 |
| H_res | (B, T, n, n) | 来自 Pre-Op 的残差混合矩阵 |

| 返回值 | 形状 | 说明 |
|---|---|---|
| x_next | (B, T, n, C) | 更新后的多流特征，作为下一层的输入 |

---
**C必须能被128整除**  

<a id="mhcop-实现状态"></a>
### mhcop 实现状态

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ❌   | ✅     | ✅      |
| JAX       | ❌   | ✅     | ✅      |
| TensorFlow| ❌   | ❌     | ✅      |
| NumPy     | ❌   | ❌     | ✅      |
| OpenVINO  | ❌   | ❌     | ✅      |

> **实现备注：**
> 1. **Torch 用户建议必开**：在 A100 上，`mhc_post_op` 相比 `torch.compile` 有约 3 倍提速，`mhc_pre_op` 约 8 倍。
> 2. **JAX 用户按需选择**：XLA 的原生性能很强，如果追求纯推理吞吐量，建议使用native模式的`mhc_pre_op`搭配triton的`mhc_post_op`。因为单算子测试中，`mhc_post_op` 相比 `torch.compile` 有约 1.1 倍提速，`mhc_pre_op` 约 2 倍，但是XLA可以在整个图上做更深度的融合。但 **Triton 版非常省显存**，在 BERT-like 或 GPT-like 深度模型中，所以我们训练建议使用完全的triton实现。 
> 3. **算子一致性**：JAX 和 Torch 共享同一套 Triton 逻辑，性能差异源于各后端对外部算子的调度开销不同（XLA 打包能力强，Torch 相对较弱）。

---

<a id="rwkv7op-使用方法"></a>
## rwkv7op 使用方法

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
    chunk_size: int = 16,
    mask=None,
):
    """
    分块 Delta Rule 注意力接口。

    Args:
        r:  [B, T, H, K]
        w:  [B, T, H, K]
        k:  [B, T, H, K]
        v:  [B, T, H, K]
        a:  [B, T, H, K]
        b:  [B, T, H, K]
        initial_state: 初始状态 [B, H, K, K] 或 [1, H, K, K]
        output_final_state: 是否返回最终状态
        head_first: 是否 head-first 格式，不支持变长
        chunk_size: chunk 长度，默认 16；cuda/triton/pallas 后端 T 必须被其整除
        mask: [B, T]，决定这个状态是否被更新，1 更新 0 不更新。注意开启这个你的训练速度会慢一倍。
              因此我更推荐 v *= mask, a *= mask, ops.where(mask, w, -1e9) 的方式来做 mask

    Returns:
        o:           输出 [B, T, H, K] 或 [B, H, T, K]
        final_state: 最终状态 [B, H, K, K] 或 None
    """
```
generalized_delta_rule_inference和generalized_delta_rule的区别是前者没有梯度。因为不需要存储激活值，所以可以节省一部分显存。

<a id="cuda-kernel-特殊用法"></a>
### cuda-kernel 特殊用法

- torch-cuda 和 jax-cuda kernel 下 `head_size` 与 `chunk_size` 都是编译期参数，默认分别为 64 和 16。  
- 若需要 `head_size ≠ 64` 或 `chunk_size ≠ 16`，请使用工厂函数：

```python
from rwkv_ops import get_generalized_delta_rule

rwkv7_op, rwkv7_op_inference = get_generalized_delta_rule(
    HEAD_SIZE=your_head_size, KERNEL_TYPE="cuda", chunk_size=32
)
```

- `chunk_size` 会作为 `-D_CHUNK_LEN_` 宏编译进 CUDA kernel，因此不同 `chunk_size` 会各自编译一次，生成独立的 `.so` / torch extension。切换 `chunk_size` 不会复用旧 kernel，避免结果错误。
- 对于 padding 处理：

```python
if padding_mask is not None:
    w += (1 - padding_mask) * -1e9
```
- 基于循环的算子可以针对 left padding 和 right padding 都能成功处理。
- 而如果用的是chunkwise算子，建议统一left padding，如果是cuda或者原生，则都left right都能正确处理


<a id="rwkv7op-实现状态"></a>
### rwkv7op 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ✅     | ✅     |
| JAX         | ✅   | ✅     | ✅¹    |
| TensorFlow  | ❌    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

> ¹ JAX 后端的 `native` 在 GPU/TPU 上为 Pallas 实现（`jax_pallas_kernel.py`），Torch 后端的 `native` 在非 CPU 平台为 Triton 实现，其余为纯 Keras ops。


---
1. `native` 为原生算子，速度慢且显存高。
2. `cuda`和 `triton`为基于 CUDA 的原生算子，速度很快，并且kernel内部使用fp32实现，所以精度也很高。缺点就是长序列的时候比较吃亏跑不满。

<a id="rwkv7_op_rnn-使用方法"></a>
## rwkv7_op_rnn 使用方法

<a id="背景"></a>
### 背景
这是RWKV7 OP的特殊情况，就是我们只考虑长度=1的情况。专门用于推理的decode阶段的加速

<a id="使用方法"></a>
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
<a id="rwkv7_op_rnn-实现状态"></a>
### rwkv7_op_rnn 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ❌     | ✅     |
| JAX         | ✅   | ❌     | ✅     |
| TensorFlow  | ❌    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

1. native实现我们直接复用了rwkv7_op的native实现
2. **这个算子没有梯度**

<a id="rwkv7op_sane-使用方法"></a>
## rwkv7op_sane 使用方法

`rwkv7op_sane` 实现 **State Anomaly Neutralization（SANE）**，一种在 chunk 边界对 RWKV-7 的 state 做软裁剪的数值稳定技术。详见论文 [SANE: State Anomaly Neutralization for RWKV-7](https://arxiv.org/pdf/2608.22354)。

```python
from rwkv_ops import generalized_delta_rule_sane, generalized_delta_rule_sane_inference

def generalized_delta_rule_sane(
    r,
    w,
    k,
    v,
    a,
    b,
    tau,                  # [B, T//chunk_size, H]，float32，已预处理为 softplus(param)+1，必须 > 0
    mask=None,            # [B, T//chunk_size]，float32，1 表示执行 SANE，0 表示跳过；所有 head 共享
    initial_state=None,
    output_final_state: bool = True,
    head_first: bool = False,
    chunk_size: int = 16,
):
    """
    带 State Anomaly Neutralization 的 RWKV-7 广义 Delta 规则（训练 / prefill 通用）。

    调度规则：
    - 当 ``output_final_state=False`` 时，内部调用无 mask 算子，chunk 边界无条件
      执行 State Anomaly Neutralization，以节省 mask 读取/分支开销。
    - 当 ``mask=None`` 且 ``output_final_state=True`` 时，同样调用无 mask 算子，
      但会弹出警告并把返回的 ``final_state`` 设为 ``None``，避免用户误用可能被
      padding 污染的 state。
    - 只有 ``output_final_state=True`` 且显式传入 ``mask`` 时，才使用带 mask 算子。

    Args:
        r, w, k, v, a, b: [B, T, H, K] 或 [B, H, T, K]，T 必须被 chunk_size 整除。
        tau: [B, T//chunk_size, H]，float32，必须 > 0。
        mask: [B, T//chunk_size]，float32，0/1 标记每个 chunk 是否执行 State Anomaly Neutralization；
              只有需要返回 final_state 且显式提供时才生效。padding chunk 请置 0，
              并配合 k=0, a=0, w=-inf。
        initial_state: [B, H, K, K] 或 [1, H, K, K]。
        output_final_state: 是否返回最终 State。
        head_first: 是否 head-first。
        chunk_size: chunk 长度，默认 16。CUDA 后端为编译期常量，需通过工厂函数指定；Triton/Pallas/native 后端可在调用时传入。

    Returns:
        out: [B, T, H, K]
        final_state: [B, H, K, K] 或 None（mask=None 且 output_final_state=True 时）
    """
```

`generalized_delta_rule_sane_inference` 与 `generalized_delta_rule_sane` 接口一致，但**不计算梯度**，可节省显存。
注意：推理 kernel 按 chunk 读取 `tau`，因此 `tau` 的长度只需等于 `T // chunk_size`，
**T 不再强制要求被 chunk_size 整除**；若需要任意长度 prefill，也可使用下方的单步 RNN 接口。

<a id="rwkv7op_sane-实现状态"></a>
### rwkv7op_sane 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ✅     | ✅     |
| JAX         | ✅   | ✅     | ✅¹    |
| TensorFlow  | ❌    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

<a id="rwkv7_op_sane_rnn-使用方法"></a>
## rwkv7_op_sane_rnn 使用方法

```python
from rwkv_ops import rwkv7_op_sane_rnn

def rwkv7_op_sane_rnn(
    r,                    # [B, 1, H, K] 或 [B, H, 1, K]
    w,
    k,
    v,
    a,
    b,
    tau,                  # [B, H]，float32
    do_sane,                # bool 或 [B] bool，True 表示本步后执行 State Anomaly Neutralization
    initial_state=None,
    output_final_state: bool = True,
    head_first: bool = False,
):
    """
    带 State Anomaly Neutralization 的 RWKV-7 单步推理（RNN 模式）。
    输出基于 SANE 前的 State，state_out 已按 do_sane 应用 SANE。
    """
```

调用示例（每 chunk_size 步触发一次 SANE，默认 16）：

```python
for step in range(seq_len):
    do_sane = (step % chunk_size == chunk_size - 1)
    out, state = rwkv7_op_sane_rnn(
        r[step], w[step], k[step], v[step], a[step], b[step],
        tau=tau, do_sane=do_sane, initial_state=state
    )
```

<a id="rwkv7_op_sane_rnn-实现状态"></a>
### rwkv7_op_sane_rnn 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ❌     | ✅     |
| JAX         | ✅   | ❌     | ✅     |
| TensorFlow  | ❌    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

1. 单步算子**没有梯度**。
2. CUDA 版本会强制把输入 cast 到 bfloat16，与 rwkv7_op_rnn 行为一致。

<a id="gdn_recurrent-使用方法"></a>
## gdn_recurrent 使用方法

`gdn_recurrent` 提供 **Gated DeltaNet** 的逐步 recurrent 实现，支持训练、推理与单步 RNN 三种入口。默认输入 layout 为 `[B, T, H, K/V]`，设置 `head_first=True` 可切换为 `[B, H, T, K/V]`。

```python
from rwkv_ops import (
    gated_delta_net_recurrent,
    gated_delta_net_recurrent_inference,
    gated_delta_net_recurrent_single_step,
)

# 训练 / prefill（可求梯度）
out, final_state = gated_delta_net_recurrent(
    q, k, v, g, beta,
    initial_state=h0,
    output_final_state=True,
    head_first=False,
)

# 推理专用（无梯度，省显存）
out, final_state = gated_delta_net_recurrent_inference(
    q, k, v, g, beta,
    initial_state=h0,
    output_final_state=True,
    head_first=False,
)

# 单步 RNN（decode 阶段）
out, state = gated_delta_net_recurrent_single_step(
    q, k, v, g, beta,
    initial_state=state,
    output_final_state=True,
    head_first=True,  # 单步目前只支持 head_first=True
)
```

<a id="函数接口说明-1"></a>
### 函数接口说明

<a id="gated_delta_net_recurrent"></a>
#### `gated_delta_net_recurrent`

| 参数 | 形状 | 说明 |
|---|---|---|
| q, k | (B, T, H, K) | 查询与键，内部先做 L2 归一化 |
| v | (B, T, H, V) | 值 |
| g | (B, T, H) | log-space 衰减门控 |
| beta | (B, T, H) | 写入强度，需已在外部过 sigmoid，落在 (0, 1) |
| initial_state | (B, H, K, V) 或 (1, H, K, V)，可选 | 初始 recurrent state |
| output_final_state | bool | 是否返回最终 state |
| head_first | bool | 输入输出是否 head 维优先 |
| chunk_size | int | chunk 长度，默认 16；纯 recurrent 实现忽略该参数（仅签名一致） |

| 返回值 | 形状 | 说明 |
|---|---|---|
| out | (B, T, H, V) | 与 `v` 同 dtype |
| final_state | (B, H, K, V) 或 None | 最终 state |

<a id="gated_delta_net_recurrent_inference"></a>
#### `gated_delta_net_recurrent_inference`

接口与 `gated_delta_net_recurrent` 完全一致，但**不计算梯度**，因此不保存反向所需的 `kv_mem` 与 `state_chkp` 等中间量，显存占用更低。

<a id="gated_delta_net_recurrent_single_step"></a>
#### `gated_delta_net_recurrent_single_step`

| 参数 | 形状 | 说明 |
|---|---|---|
| q, k | (B, H, K) | 单步查询与键 |
| v | (B, H, V) | 单步值 |
| g | (B, H) | 单步 log-space 衰减门控 |
| beta | (B, H) | 单步写入强度，已 sigmoid |
| initial_state | (B, H, K, V) 或 (1, H, K, V)，可选 | 当前 state |
| output_final_state | bool | 是否返回下一步 state |
| head_first | bool | 单步目前只支持 `head_first=True` |
| chunk_size | int | chunk 长度，默认 16；单步实现忽略该参数（仅签名一致） |

| 返回值 | 形状 | 说明 |
|---|---|---|
| out | (B, H, V) | 与 `v` 同 dtype |
| next_state | (B, H, K, V) | 下一步 state |

<a id="gdn_recurrent-实现状态"></a>
### gdn_recurrent 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ✅     | ✅     |
| JAX         | ❌   | ✅     | ✅¹    |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

> ¹ JAX 后端的 `native` 在 GPU/TPU 上为 Pallas 实现（`jax_pallas_kernel.py`），`triton` 显式 `KERNEL_TYPE="triton"` 时为 JAX-Triton 实现；其余为纯 Keras ops。

1. 训练入口支持反向传播；推理与单步入口**没有梯度**。
2. PyTorch `cuda` 后端的 `chunk_size` 作为编译期常量按 `(K, V, chunk_size)` 在首次调用时懒编译，调用时可传入不同 `chunk_size`（各自编译一次）。
3. chunkwise 版本位于 `gdn_chunk/`，见下节。

<a id="gdn_recurrent_sane-使用方法"></a>
## gdn_recurrent_sane 使用方法

`gdn_recurrent_sane` 在 `gdn_recurrent` 基础上加入 **State Anomaly Neutralization（SANE）**：在每个 chunk 边界对 state 做 `state = tau * tanh(state / tau)` 软裁剪，用于抑制 padding chunk 或异常状态污染。支持训练、推理与单步 RNN 三种入口。

```python
from rwkv_ops import (
    gated_delta_net_recurrent_sane,
    gated_delta_net_recurrent_sane_inference,
    gated_delta_net_recurrent_sane_single_step,
)

# 训练 / prefill（可求梯度）
out, final_state = gated_delta_net_recurrent_sane(
    q, k, v, g, beta, tau, mask=mask,
    initial_state=h0,
    output_final_state=True,
    head_first=False,
)

# 推理专用（无梯度，省显存）
out, final_state = gated_delta_net_recurrent_sane_inference(
    q, k, v, g, beta, tau, mask=mask,
    initial_state=h0,
    output_final_state=True,
    head_first=False,
)

# 单步 RNN（decode 阶段）
out, state = gated_delta_net_recurrent_sane_single_step(
    q, k, v, g, beta, tau, do_sane,
    initial_state=state,
    output_final_state=True,
    head_first=True,  # 单步目前只支持 head_first=True
)
```

<a id="函数接口说明-2"></a>
### 函数接口说明

<a id="gated_delta_net_recurrent_sane"></a>
#### `gated_delta_net_recurrent_sane`

| 参数 | 形状 | 说明 |
|---|---|---|
| q, k | (B, T, H, K) | 查询与键，内部先做 L2 归一化 |
| v | (B, T, H, V) | 值 |
| g | (B, T, H) | log-space 衰减门控 |
| beta | (B, T, H) | 写入强度，需已在外部过 sigmoid，落在 (0, 1) |
| tau | (B, T//chunk_size, H) | SANE 阈值，必须 > 0 |
| mask | (B, T//chunk_size)，可选 | >0 的 chunk 边界执行 SANE；仅当 `output_final_state=True` 时生效 |
| initial_state | (B, H, K, V) 或 (1, H, K, V)，可选 | 初始 recurrent state |
| output_final_state | bool | 是否返回最终 state |
| head_first | bool | 输入输出是否 head 维优先 |
| chunk_size | int | SANE chunk 长度，默认 16；决定 `tau`/`mask` 的 chunk 维度 |

| 返回值 | 形状 | 说明 |
|---|---|---|
| out | (B, T, H, V) | 与 `v` 同 dtype，基于 SANE 之前的 state |
| final_state | (B, H, K, V) 或 None | 最终 state；`mask=None` 且 `output_final_state=True` 时为 None 并报警告 |

<a id="gated_delta_net_recurrent_sane_inference"></a>
#### `gated_delta_net_recurrent_sane_inference`

接口与 `gated_delta_net_recurrent_sane` 基本一致，但**不计算梯度**，不保存反向所需的 `kv_mem` 与 `state_chkp` 等中间量，显存占用更低。支持任意长度 `T`。

<a id="gated_delta_net_recurrent_sane_single_step"></a>
#### `gated_delta_net_recurrent_sane_single_step`

| 参数 | 形状 | 说明 |
|---|---|---|
| q, k | (B, H, K) | 单步查询与键 |
| v | (B, H, V) | 单步值 |
| g | (B, H) | 单步 log-space 衰减门控 |
| beta | (B, H) | 单步写入强度，已 sigmoid |
| tau | (B, H) | 单步 SANE 阈值 |
| do_sane | (B,) | >0 执行 SANE，否则跳过 |
| initial_state | (B, H, K, V) 或 (1, H, K, V)，可选 | 当前 state |
| output_final_state | bool | 是否返回下一步 state |
| head_first | bool | 单步目前只支持 `head_first=True` |
| chunk_size | int | chunk 长度，默认 16；单步实现忽略该参数（仅签名一致） |

| 返回值 | 形状 | 说明 |
|---|---|---|
| out | (B, H, V) | 与 `v` 同 dtype |
| next_state | (B, H, K, V) | 下一步 state |

<a id="gdn_recurrent_sane-实现状态"></a>
### gdn_recurrent_sane 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ❌   | ✅     | ✅     |
| JAX         | ❌   | ✅     | ✅¹    |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

> ¹ JAX 后端的 `native` 在 GPU/TPU 上为 Pallas 实现（`jax_pallas_kernel.py`），`triton` 显式 `KERNEL_TYPE="triton"` 时为 JAX-Triton 实现；其余为纯 Keras ops。

1. 训练入口支持反向传播（含 `tau` 梯度）；推理与单步入口**没有梯度**。
2. `mask=None` 时仍执行无条件 SANE，但 `output_final_state=True` 会发出 `UserWarning` 并将 `final_state` 置为 `None`，避免 padding 污染被误用。

<a id="gdn_chunk-使用方法"></a>
## gdn_chunk 使用方法

`gdn_chunk` 提供 **Gated DeltaNet** 的分块并行（chunkwise）训练实现。默认输入 layout 为 `[B, T, H, K/V]`；内部会转置为 `[B, H, T, K/V]` 以匹配 Triton kernel。

```python
from rwkv_ops import gated_delta_net_chunk

out, final_state = gated_delta_net_chunk(
    q, k, v, g, beta,
    initial_state=h0,
    output_final_state=True,
    chunk_size=16,
)
```

<a id="函数接口说明-3"></a>
### 函数接口说明

<a id="gated_delta_net_chunk"></a>
#### `gated_delta_net_chunk`

| 参数 | 形状 | 说明 |
|---|---|---|
| q, k | (B, T, H, K) | 查询与键，内部先做 L2 归一化 |
| v | (B, T, H, V) | 值 |
| g | (B, T, H) | log-space 衰减门控 |
| beta | (B, T, H) | 写入强度，需已在外部过 sigmoid，落在 (0, 1) |
| initial_state | (B, H, K, V) 或 (1, H, K, V)，可选 | 初始 recurrent state |
| output_final_state | bool | 是否返回最终 state |
| chunk_size | int | chunk 长度，必须整除 T |

| 返回值 | 形状 | 说明 |
|---|---|---|
| out | (B, T, H, V) | 与 `v` 同 dtype |
| final_state | (B, H, K, V) 或 None | 最终 state |

<a id="gdn_chunk-实现状态"></a>
### gdn_chunk 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ❌   | ✅     | ✅     |
| JAX         | ❌   | ✅     | ✅     |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

> 训练入口支持反向传播。JAX 侧 `triton` 需显式 `KERNEL_TYPE="triton"` 并安装 `jax-triton`。

---

<a id="gdn_chunk_sane-使用方法"></a>
## gdn_chunk_sane 使用方法

`gdn_chunk_sane` 在 `gdn_chunk` 基础上加入 **State Anomaly Neutralization（SANE）**：在每个 chunk 边界对跨 chunk 传递的 state 执行 `state = tau * tanh(state / tau)`，并按 `mask` 选择是否生效。其余行为与 `gdn_chunk` 一致。

```python
from rwkv_ops import gated_delta_net_chunk_sane

out, final_state = gated_delta_net_chunk_sane(
    q, k, v, g, beta, tau,
    mask=mask,                 # [B, T//chunk_size]，可选
    initial_state=h0,
    output_final_state=True,
    chunk_size=16,
)
```

<a id="函数接口说明-4"></a>
### 函数接口说明

<a id="gated_delta_net_chunk_sane"></a>
#### `gated_delta_net_chunk_sane`

| 参数 | 形状 | 说明 |
|---|---|---|
| q, k | (B, T, H, K) | 查询与键，内部先做 L2 归一化 |
| v | (B, T, H, V) | 值 |
| g | (B, T, H) | log-space 衰减门控 |
| beta | (B, T, H) | 写入强度，需已在外部过 sigmoid，落在 (0, 1) |
| tau | (B, T//chunk_size, H) | SANE 阈值，必须 > 1 |
| mask | (B, T//chunk_size)，可选 | >0 的 chunk 边界执行 SANE；为 None 时无条件 SANE |
| initial_state | (B, H, K, V) 或 (1, H, K, V)，可选 | 初始 recurrent state |
| output_final_state | bool | 是否返回最终 state；mask=None 时强制为 None 并发出警告 |
| chunk_size | int | chunk 长度，必须整除 T |

| 返回值 | 形状 | 说明 |
|---|---|---|
| out | (B, T, H, V) | 与 `v` 同 dtype |
| final_state | (B, H, K, V) 或 None | 最终 state；mask=None 时为 None |

<a id="gdn_chunk_sane-实现状态"></a>
### gdn_chunk_sane 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ❌   | ✅     | ✅     |
| JAX         | ❌   | ✅     | ✅     |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

> 训练入口支持反向传播（含 `tau` 梯度）。JAX 侧 `triton` 需显式 `KERNEL_TYPE="triton"` 并安装 `jax-triton`。
> `mask=None` 时输出仍使用无条件 SANE，但 `output_final_state=True` 会发出 `UserWarning` 并将 `final_state` 置为 `None`。

<a id="分布式并行"></a>
## 分布式并行（JAX）

JAX 侧所有加速算子都通过 `custom_partitioning` + einsum 风格 `sharding_rule`
声明了分片传播规则，可在 `jax.jit` + `NamedSharding` 下自动正确分区，支持
**DP（batch 维并行）** 与 **TP（head 维并行）**：

| 算子（jax） | DP (batch) | TP (head) |
|---|---|---|
| rwkv7 cuda / triton / pallas | ✅ | ✅ |
| rwkv7_sane cuda / triton / pallas | ✅ | ✅ |
| rwkv7 / rwkv7_sane 单步 cuda | ✅ | ✅ |
| gdn_recurrent triton / pallas | ✅ | ✅ |
| gdn_recurrent_sane triton / pallas | ✅ | ✅ |
| gdn_chunk triton | ✅ | ✅ |
| gdn_chunk_sane triton | ✅ | ✅ |
| rwkv6 cuda | ✅ | ❌ |

> rwkv6 的 channel 维（C = H × N）在分片规则中是一个整体，head 维未暴露，
> 因此只支持按 batch 的数据并行；需要 TP 请使用 rwkv7 / rwkv7_sane。

**只允许切 batch 或 head 维**；切 time 或 head_size 维会得到错误结果
（扫描需要完整 T，state 需要完整 head_size）。SANE 的 `tau` 带 head 维
（TP 可切）、`mask` 无 head 维（TP 下自动复制）。

使用示例（TP：沿 head 维切分）：

```python
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

mesh = Mesh(jax.devices(), ("h",))
with jax.set_mesh(mesh):
    q_shd = NamedSharding(mesh, PartitionSpec(None, None, "h", None))
    s_shd = NamedSharding(mesh, PartitionSpec(None, "h", None, None))
    y, state = jax.jit(
        lambda *x: rwkv7_op(r=x[0], w=x[1], k=x[2], v=x[3], a=x[4], b=x[5],
                            initial_state=x[6]),
        in_shardings=(q_shd,) * 6 + (s_shd,),
    )(r, w, k, v, a, b, h0)
```

DP 时把输入的 batch 维切到 mesh 轴上即可（规则同样覆盖反向传播与
state checkpoint / final_state 的分片传播）。推理侧的 DP 通常是进程级
副本，不经 mesh；单步算子的 TP 规则面向 TP 推理部署。

---

<a id="rwkv6op-使用方法"></a>
## rwkv6op 使用方法

<a id="pytorch-使用注意事项"></a>
### PyTorch 使用注意事项

- 安装依赖：`keras`、`ninja`、完整的 CUDA 工具包。
- 若使用 VS Code + 虚拟环境调试，请务必在终端手动激活虚拟环境，再运行代码，否则 ninja 可能无法工作。
- 虽然 PyTorch 在「虚拟环境中的 CUDA 版本」与「全局 CUDA 版本」不一致时仍可正常运行，但强烈建议保持一致。
- 算子线程安全（无状态），可在多处调用。

<a id="jax-使用注意事项"></a>
### JAX 使用注意事项

- 安装依赖：`keras`、`cmake`、`gcc`、完整的 CUDA 工具包。
- 即使通过虚拟环境为 JAX 安装 CUDA，也必须在系统级安装完整 CUDA；两者版本需一致，以保证 JAX 并行编译速度。
- JAX 编译依赖 `/usr/local/cuda` 软链接，如不存在请手动创建：
  ```shell
  sudo ln -sf /usr/local/cuda-12.4 /usr/local/cuda
  ```
- 确保 `nvcc -V` 正常输出，且 `which nvcc` 指向正确版本。
- JAX `cuda` 后端基于 `jax.ffi`，支持 JAX >= 0.4.31（含 0.6.x）。CUDA 路径仅对 `bfloat16` 加速；非 `bfloat16` 输入会发出警告并强制 cast 为 `bfloat16`，不再回退到 `native`。

<a id="tensorflow-使用注意事项"></a>
### TensorFlow 使用注意事项

- 仅提供基于原生 API 的 `RWKV6` 算子，效率较低。

---

<a id="使用方法-1"></a>
### 使用方法

RWKV-6 现在与 RWKV-7 一样提供**函数式接口**。

```python
from rwkv_ops import rwkv6_op  # 或兼容别名 RWKV6_OP

y, final_state = rwkv6_op(
    r, k, v, w, u,
    initial_state=None,         # 可选初始状态
    output_final_state=False,   # 是否返回结束状态
    state_map=None,             # 状态映射，见下文
    head_first=False,           # 输入是否为 [B, H, T, N]
)
```

| 参数 | 形状 | 说明 |
|---|---|---|
| r, k, v, w | (B, T, C) 或 (B, H, T, N) | — |
| u | (H, N) 或 (C,) | — |
| initial_state | (S, H, N, N) 或 (H, N, N) | S=1 时所有样本共用；S=B 时一一对应 |
| state_map | (B,) int64 | 指定每个样本用到的 initial_state 索引 |

| 返回值 | 形状 | 说明 |
|---|---|---|
| y | (B, T, C) 或 (B, H, T, N) | 输出 |
| final_state | (B, H, N, N) 或 None | 结束状态 |

> 如需指定非默认的 `head_size` 或 `max_sequence_length`，请使用：
> ```python
> from rwkv_ops import get_rwkv6_kernel
> rwkv6_op = get_rwkv6_kernel(HEAD_SIZE=64, KERNEL_TYPE="cuda", MAX_SEQUENCE_LENGTH=4096)
> ```
> `MAX_SEQUENCE_LENGTH` 是 CUDA kernel 的编译期常量 `_T_`，必须大于等于实际序列长度；`native` 实现会忽略该参数。

---


<a id="rwkv6op-实现状态"></a>
### rwkv6op 实现状态

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ❌     | ✅     |
| JAX         | ✅   | ❌     | ✅     |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |

JAX `cuda` 后端基于 `jax.ffi`，支持 JAX >= 0.4.31（含 0.6.x）。CUDA 路径仅对 `bfloat16` 加速；非 `bfloat16` 输入会发出警告并强制 cast 为 `bfloat16`，不再回退到 `native`。


<a id="工厂函数与自定义参数"></a>
## 工厂函数与自定义参数

包入口 `from rwkv_ops import *` 会按环境变量实例化一组**默认算子**：`HEAD_SIZE=64`、`chunk_size=16`、`KERNEL_TYPE` 取自 `KERNEL_TYPE` 环境变量（默认 `cuda`）。

若需要自定义 `head_size`、`chunk_size` 或 `KERNEL_TYPE`，请使用对应的 `get_*` 工厂函数：

```python
from rwkv_ops import (
    get_generalized_delta_rule,           # RWKV-7
    get_generalized_delta_rule_sane,      # RWKV-7-SANE
    get_gated_delta_net_chunk,            # GDN chunkwise
    get_gated_delta_net_chunk_sane,       # GDN chunkwise SANE
    get_gated_delta_net_recurrent,        # GDN recurrent（无 SANE）
    get_gated_delta_net_recurrent_sane,   # GDN recurrent SANE
)

# RWKV-7 CUDA：HEAD_SIZE 与 chunk_size 均为编译期常量，必须在工厂指定
rwkv7_op, rwkv7_op_inference = get_generalized_delta_rule(
    HEAD_SIZE=64, KERNEL_TYPE="cuda", chunk_size=32
)

# RWKV-7-SANE CUDA：同样通过工厂指定 chunk_size
rwkv7_op_sane, rwkv7_op_sane_inference = get_generalized_delta_rule_sane(
    HEAD_SIZE=64, KERNEL_TYPE="cuda", chunk_size=32
)

# Gated DeltaNet chunkwise（triton/native 可在调用时传 chunk_size）
gdn_chunk = get_gated_delta_net_chunk(KERNEL_TYPE="triton", chunk_size=32)

# Gated DeltaNet chunkwise SANE（triton/native 可在调用时传 chunk_size）
gdn_chunk_sane = get_gated_delta_net_chunk_sane(
    KERNEL_TYPE="triton", chunk_size=32
)

# Gated DeltaNet recurrent（无 SANE；chunk_size 仅签名一致，可忽略）
gdn_recurrent = get_gated_delta_net_recurrent(KERNEL_TYPE="triton", chunk_size=32)

# Gated DeltaNet recurrent SANE（triton/native 可在调用时传 chunk_size）
gdn_recurrent_sane = get_gated_delta_net_recurrent_sane(
    KERNEL_TYPE="triton", chunk_size=32
)
```

**注意：**
- **CUDA 后端**：`chunk_size` 会作为 `-D_CHUNK_LEN_` 宏编译进 kernel，因此必须通过工厂函数指定；返回的算子仍带 `chunk_size` 参数，但传入与编译值不同的 `chunk_size` 会报错。不同 `chunk_size` 会各自编译一次，互不影响。涉及算子：`generalized_delta_rule`、`generalized_delta_rule_sane`、`gated_delta_net_chunk`、`gated_delta_net_chunk_sane`、`gated_delta_net_recurrent_sane`。
- **Triton / Pallas / native 后端**：`chunk_size` 可在每次调用时传入，工厂函数仅设定默认值；修改 `chunk_size` 不会触发重新编译。
- `chunk_size` 默认 16；`gated_delta_net_recurrent`（无 SANE）接受但忽略该参数，仅保持签名一致。
- 工厂函数返回的算子签名与默认算子一致，可直接调用。


<a id="测试"></a>
## 测试

项目已迁移到 pytest，测试按后端隔离：

```bash
# 安装测试依赖
pip install -e ".[test]"

# torch 后端（包含 rwkv6/rwkv7 CUDA、推理、单步、triton、mHC）
pytest tests/torch -v

# jax 后端（需要 cmake 与兼容的 GCC）
pytest tests/jax -v

# numpy / tensorflow 只做 native 简单烟雾测试
pytest tests/numpy -v
pytest tests/tensorflow -v

# 跳过编译较重的 slow 测试
pytest tests/torch tests/jax -v -m "not slow"
```

每次 pytest 会话结束后会自动清理 `build_*`、`.so` 与 `__pycache__`。

> **注意**：不同后端目录中的测试文件采用了不同的模块名（如 `test_torch_rwkv6.py` / `test_jax_rwkv6.py`），以保证从 `tests` 根目录直接收集时不会出现 `import file mismatch`。如果你新增跨后端测试，请保持文件名唯一。
>
> **JAX CUDA 编译兼容性**：`rwkv_ops` 会自动使用项目内的 `rwkv_ops/cuda_tools/nvcc_wrap` 绕过 glibc 2.41+ 与 CUDA 13.1 之间 `rsqrt` 头文件声明冲突，无需手动修改系统 CUDA 头文件。
