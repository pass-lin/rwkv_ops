# RWKV OPS Project

> RWKV will keep evolving, so the core operators will be updated accordingly.  
> This repository only maintains the **operators**, not the layers or models.  
> We aim to provide GPU operators for as many frameworks as possible.  

### Current Support
| Operator Type | Frameworks |
|---------------|------------|
| GPU Operators | PyTorch, JAX (TensorFlow will be added once Google supports Triton) |
| Native (CPU) Operators | PyTorch, JAX, TensorFlow, NumPy |

> Future support for MLX or OpenVINO may be added if the Keras ecosystem expands.  
> Note: this package depends on `keras`.

---

## Installation

```bash
pip install rwkv_ops
```

---

## Environment Variables

| Variable | Description | Allowed Values | Default | Priority |
|---|---|---|---|---|
| `KERAS_BACKEND` | Keras backend | `jax`, `torch`, `tensorflow`, `numpy` | — | Low |
| `KERNEL_BACKEND` | Operator backend | `jax`, `torch`, `tensorflow`, `numpy` | `torch` | **High** |
| `KERNEL_TYPE` | Implementation type | `triton`, `cuda`, `native` | — | — |

> If `KERNEL_BACKEND` is set, it will be used directly; otherwise, fall back to `KERAS_BACKEND`; if both are empty, `torch` is used.  
> `native` means pure CPU operators without chunking—slow and memory-hungry.

---

## rwkv7op Usage

```python
from rwkv_ops import generalized_delta_rule  # or: from rwkv_ops import rwkv7_op, same thing

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
    Chunked Delta-Rule attention interface.

    Args:
        q:  [B, T, H, K]
        k:  [B, T, H, K]
        v:  [B, T, H, V]
        a:  [B, T, H, K]
        b:  [B, T, H, K]
        gk: [B, T, H, K]  # decay term in log space!
        initial_state: Initial state [N, H, K, V], N = number of sequences
        output_final_state: Whether to return the final state
        head_first: Whether input is head-first; not compatible with variable-length sequences

    Returns:
        o:           Output [B, T, H, V] or [B, H, T, V]
        final_state: Final state [N, H, K, V] or None
    """
```

### torch-cuda Special Case

- In torch-cuda, `head_size` is a kernel parameter and defaults to 64.  
- If your `head_size ≠ 64`, use:

```python
from rwkv_ops import get_generalized_delta_rule

generalized_delta_rule, RWKV7_USE_KERNEL = get_generalized_delta_rule(
    your_head_size, KERNEL_TYPE="cuda"
)
```

- `RWKV7_USE_KERNEL` is a constant flag indicating whether the chunked kernel is used.  
- Padding logic differs:

```python
if padding_mask is not None:
    if RWKV7_USE_KERNEL:
        w += (1 - padding_mask) * -1e9
    else:
        w = w * padding_mask + 1 - padding_mask
```

---

### rwkv7op Implementation Status

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ✅     | ✅     |
| JAX         | ❌   | ✅     | ✅     |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |

---

## rwkv6op Usage

### PyTorch Notes

- Dependencies: `keras`, `ninja`, full CUDA toolkit.
- If using VS Code + virtual env, **manually activate** the environment in the terminal before running; otherwise `ninja` may fail.
- Even if the CUDA version inside the venv differs from the system one, the operator still works, but keeping them identical is strongly recommended.
- Due to PyTorch limitations, **only one** `RWKV6_OP` instance per process is allowed.  
  The operator is stateless and thread-safe, so you can call it from multiple places.

### JAX Notes

- Dependencies: `keras`, `gcc`, `pybind11`, full CUDA toolkit.
- Even if JAX is installed in a venv with CUDA, a system-wide CUDA installation is required and versions must match for faster parallel compilation.
- JAX compilation relies on the symlink `/usr/local/cuda`. Create it if missing:
  ```shell
  sudo ln -sf /usr/local/cuda-12.4 /usr/local/cuda
  ```
- Ensure `nvcc -V` prints correctly and `which nvcc` points to the expected version.
- Due to JAX limitations, **only one** `RWKV6_OP` instance per process is allowed.  
  The operator is stateless and thread-safe.
- JAX ≥ 0.6.0 no longer uses CUDA kernels; native kernels are used instead.  
  Recommended JAX version: 0.4.34.

### TensorFlow Notes

- Only a native-API-based RWKV6 operator is provided; it is for inference only and slower.

---

### API Reference

```python
from rwkv_ops import RWKV6_OP

operator = RWKV6_OP(
    head_size=64,               # head dimension, use 64 if unsure
    max_sequence_length=4096,   # max length during training; inference can be longer
    ops_loop=False              # optional: fall back to high-level impl when seq_len=1
)
```

#### Call Signature

```python
y, y_state = operator(
    r, k, v, w, u,
    with_state=False,  # enable custom initial state / return final state
    init_state=None,   # initial state [n_state, num_heads, head_size, head_size]
    state_map=None     # int32 1-D array mapping batch entries to init_state indices
)
```

| Arg | Shape | Notes |
|---|---|---|
| r, k, v, w | (batch_size, seq_len, hidden_size) | — |
| u | (num_heads, head_size) or (hidden_size,) | — |
| init_state | (n_state, num_heads, head_size, head_size) | n_state=1 → shared; n_state=batch_size → per-sample |
| state_map | (batch_size,) | indices into init_state |

| Return | Shape | Notes |
|---|---|---|
| y | (batch_size, seq_len, hidden_size) | output |
| y_state | (batch_size, num_heads, head_size, head_size) or None | final state |

---

### Distributed Tips (JAX example)

- The operator itself is not distributed; PyTorch’s multi-thread distributed wrappers work out of the box.  
- For JAX, wrap the operator with `shard_map`:

```python
import os
os.environ['KERAS_BACKEND'] = 'jax'

import jax, jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from functools import partial
from rwkv_ops import RWKV6_OP

batch_size, seq_len = 24, 512
head_size, num_heads = 64, 32
hidden_size = head_size * num_heads

mesh = Mesh(jax.devices('gpu'), axis_names=('device_axis',))
device_ns = NamedSharding(mesh, P('device_axis'))

operator = RWKV6_OP(head_size=head_size, max_sequence_length=seq_len)

@partial(shard_map,
         mesh=mesh,
         in_specs=(P('device_axis'),) * 5,
         out_specs=(P('device_axis'), P('device_axis')),
         check_rep=False)
def call_kernel(r, k, v, w, u):
    # remove device dimension
    r, k, v, w, u = map(jnp.squeeze, (r, k, v, w, u))
    y, ys = operator(r, k, v, w, u, with_state=True)
    return jnp.expand_dims(y, 0), jnp.expand_dims(ys, 0)

# build inputs on devices
keys = jax.random.split(jax.random.PRNGKey(0), 5)
shapes = [(mesh.size, batch_size, seq_len, hidden_size)] * 4 + [(mesh.size, hidden_size)]
inputs = [jax.random.normal(k, s) for k, s in zip(keys, shapes)]
inputs_r, inputs_k, inputs_v, inputs_w, inputs_u = map(lambda x: jax.device_put(x, device_ns), inputs)
inputs_u = inputs_u[:, 0]  # (devices, hidden_size)

# optionally: jax.jit(call_kernel, ...)
outputs_y, y_state = call_kernel(inputs_r, inputs_k, inputs_v, inputs_w, inputs_u)

print(outputs_y.shape, outputs_y.sharding)
print(y_state.shape, y_state.sharding)
```

---

### rwkv6op Implementation Status

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ❌     | ✅     |
| JAX         | ⚠️   | ❌     | ✅     |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |

⚠️ JAX CUDA kernels only for versions < 0.6.0; recommended 0.4.34.
