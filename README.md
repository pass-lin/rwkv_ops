[English Document](ENREADME.md)
# RWKV OPS 项目
> 由于 RWKV 将持续迭代，核心算子会随之更新。  
> 本仓专门维护「算子」本身，不维护 layer 与 model；尽可能提供各框架的 GPU 算子。  
> 目前：  
> • GPU 算子：PyTorch、JAX（TensorFlow 待 Google 支持 Triton 后上线）  
> • 原生算子：PyTorch、JAX、TensorFlow、NumPy  
> 未来若 Keras 生态扩展，可能支持 MLX、OpenVINO。  
> 注意：本库依赖 `keras`。

---
## 安装方法
pip install rwkv_ops
## 环境变量

| 变量名 | 含义 | 取值 | 默认值 | 优先级 |
|---|---|---|---|---|
| `KERAS_BACKEND` | Keras 后端 | jax / torch / tensorflow / numpy | — | 低 |
| `KERNEL_BACKEND` | 算子后端 | jax / torch / tensorflow / numpy | torch | **高** |
| `KERNEL_TYPE` | 实现类型 | triton / cuda / native | — | — |

> 若 `KERNEL_BACKEND` 有值，直接采用；若为空，则用 `KERAS_BACKEND`；两者皆空则默认 torch。  
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

---


torch-cuda下head-size也是一个kernel参数，默认是64.
若 head-size ≠ 64，请使用：

```python
from rwkv_ops import get_generalized_delta_rule

generalized_delta_rule, RWKV7_USE_KERNEL = get_generalized_delta_rule(
    your_head_size, KERNEL_TYPE="cuda"
)
```

`RWKV7_USE_KERNEL` 为常量，标记是否使用 chunkwise 算子；
因为两者padding 处理逻辑不同，具体如下

```python
if padding_mask is not None:
    if RWKV7_USE_KERNEL:
        w += (1 - padding_mask) * -1e9
    else:
        w = w * padding_mask + 1 - padding_mask
```

---

### rwkv7op的实现状态


| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ✅     | ✅     |
| JAX         | ❌   | ✅     | ✅     | 
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     | 