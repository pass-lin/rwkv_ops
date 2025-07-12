
# RWKV OPS Project
> As RWKV keeps evolving, its core operators will be updated accordingly.  
> This repository is dedicated to maintaining the **operators** themselves, **not** layers or models.  
> We aim to provide GPU operators for every major framework.  
> Currently:  
> • GPU operators: PyTorch, JAX (TensorFlow will follow once Google adds Triton support)  
> • Native operators: PyTorch, JAX, TensorFlow, NumPy  
> If the Keras ecosystem expands in the future, we may add support for MLX and OpenVINO.  
> **Note**: This library depends on `keras`.

---

## Environment Variables

| Variable        | Meaning            | Values                             | Default | Priority |
|-----------------|--------------------|------------------------------------|---------|----------|
| `KERAS_BACKEND` | Keras backend      | jax / torch / tensorflow / numpy   | —       | Low      |
| `KERNEL_BACKEND`| Operator backend   | jax / torch / tensorflow / numpy   | torch   | **High** |
| `KERNEL_TYPE`   | Implementation type| triton / cuda / native             | —       | —        |

> If `KERNEL_BACKEND` is set, it is used directly.  
> If not, fall back to `KERAS_BACKEND`.  
> If neither is set, the default is `torch`.  
> `native` uses pure Python/NumPy, no chunk-wise kernel—slower and higher memory usage.

---

## rwkv7op Usage

```python
from rwkv_ops import generalized_delta_rule  # or: from rwkv_ops import rwkv7_op, identical

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
    Chunk-wise Delta-Rule attention interface.

    Args:
        q:  [B, T, H, K]
        k:  [B, T, H, K]
        v:  [B, T, H, V]
        a:  [B, T, H, K]
        b:  [B, T, H, K]
        gk: [B, T, H, K]  # decay term in log space!
        initial_state: initial state [N, H, K, V], N = batch size
        output_final_state: whether to return the final state
        head_first: whether the tensor is in head-first layout; not compatible with variable length

    Returns:
        o:           output [B, T, H, V] or [B, H, T, V]
        final_state: final state [N, H, K, V] or None
    """
```

---

When using **torch-cuda**, `head_size` is a compile-time kernel parameter and defaults to **64**.  
For any other head size, import the operator dynamically:

```python
from rwkv_ops import get_generalized_delta_rule

generalized_delta_rule, RWKV7_USE_KERNEL = get_generalized_delta_rule(
    your_head_size, KERNEL_TYPE="cuda"
)
```

`RWKV7_USE_KERNEL` is a compile-time constant indicating whether the chunk-wise kernel is used.  
Padding logic differs between the two paths:

```python
if padding_mask is not None:
    if RWKV7_USE_KERNEL:
        w += (1 - padding_mask) * -1e9
    else:
        w = w * padding_mask + 1 - padding_mask
```

---

### Implementation Status of rwkv7op

| Framework   | cuda | triton | native | NVIDIA | AMD |
|-------------|------|--------|--------|--------|-----|
| PyTorch     | ✅   | ✅     | ✅     | ✅     | ✅  |
| JAX         | ❌   | ✅     | ✅     | ✅     | ❌  |
| TensorFlow  | ❌   | ❌     | ✅     | ✅     | ❌  |
| NumPy       | ❌   | ❌     | ✅     | ✅     | ❌  |
