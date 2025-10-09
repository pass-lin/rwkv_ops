# RWKV OPS Project

> Since RWKV will continue to iterate, the core operators will be updated accordingly.  
> This repository is dedicated to maintaining the **operators** themselves, not the layers or models; it aims to provide GPU operators for various frameworks.

### Current Support

| Operator Type | Framework Support |
|---------------|-------------------|
| GPU Operators | PyTorch, JAX      |
| Native Operators | PyTorch, JAX, TensorFlow, NumPy |

> In the future, if the Keras ecosystem expands, support for MLX and OpenVINO may be added.  
> Note: This library depends on `keras`.

---

## Installation

```bash
pip install rwkv_ops
```

You also can install from source
```bash
git clone https://github.com/pass-lin/rwkv_ops.git
cd rwkv_ops
bash install.sh
```

---

## Environment Variables

| Variable Name | Meaning | Values | Default Value | Priority |
|---------------|---------|--------|---------------|----------|
| `KERAS_BACKEND` | Keras backend | `jax` / `torch` / `tensorflow` / `numpy` | — | Low |
| `KERNEL_BACKEND` | Operator backend | `jax` / `torch` / `tensorflow` / `numpy` | `torch` | **High** |
| `KERNEL_TYPE` | Implementation type | `triton` / `cuda` / `native` | `cuda` | — |

> If `KERNEL_BACKEND` is set, it will be used directly; if not, `KERAS_BACKEND` will be used; if neither is set, the default is `torch`.

---

## Usage of `rwkv7op`

```python
from rwkv_ops import generalized_delta_rule  # or from rwkv_ops import rwkv7_op, which is equivalent

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
    Chunked Delta Rule Attention Interface.

    Args:
        q:  [B, T, H, K]
        k:  [B, T, H, K]
        v:  [B, T, H, V]
        a:  [B, T, H, K]
        b:  [B, T, H, K]
        gk: [B, T, H, K]  # decay term in log space!
        initial_state: Initial state [N, H, K, V], where N is the number of sequences
        output_final_state: Whether to return the final state
        head_first: Whether in head-first format, variable-length is not supported

    Returns:
        o:           Output [B, T, H, V] or [B, H, T, V]
        final_state: Final state [N, H, K, V] or None
    """
```

### Special Usage for `torch-cuda`

- Under `torch-cuda`, `head_size` is also a kernel parameter, defaulting to 64.  
- If `head_size ≠ 64`, please use:

```python
from rwkv_ops import get_generalized_delta_rule

generalized_delta_rule, USE_TRITON_KERNEL = get_generalized_delta_rule(
    your_head_size, KERNEL_TYPE="cuda"
)
```

- `USE_TRITON_KERNEL` is a constant indicating whether the chunkwise operator is used.  
- The padding handling logic is different for the two:

```python
if padding_mask is not None:
    w += (1 - padding_mask) * -1e9
```

- For the above code, operators based on loops can handle both left padding and right padding successfully.
- However, if using the chunkwise operator, it is recommended to use left padding uniformly. If using CUDA or native, both left and right padding can be handled correctly.

### Implementation Status of `rwkv7op`

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ✅     | ✅     |
| JAX         | ✅   | ✅     | ✅     |
| TensorFlow  | ⚠️   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |

---

> `native` refers to native operators, which do not use chunkwise algorithms, are slow, and have high memory usage.
> `triton` uses chunkwise algorithms, which are fast and highly parallel, but have poor precision—use at your own risk.
> `cuda` refers to native operators based on CUDA, which are very fast and implemented in fp32 internally, ensuring high precision. However, they may struggle with long sequences.
> Tensorflow CUDA kernel only support Forward,not get graident.This implement relies on jax cuda kernel.So you should make sure you can work at jax cuda kernel.

## Usage of `rwkv6op`

### PyTorch Usage Notes

- Dependencies: `keras`, `ninja`, and a complete CUDA toolkit.
- If using VS Code with a virtual environment for debugging, make sure to manually activate the virtual environment in the terminal before running the code; otherwise, ninja may not work.
- Although PyTorch can run normally even if the CUDA version in the virtual environment is inconsistent with the global CUDA version, it is strongly recommended to keep them consistent.
- PyTorch Limitations: Only one `RWKV6_OP` object can be instantiated within the same program; the operator is thread-safe (stateless) and can be called from multiple places.

### JAX Usage Notes

- Dependencies: `keras`, `gcc`, `pybind11`, and a complete CUDA toolkit.
- Even if CUDA is installed for JAX via a virtual environment, a complete CUDA installation at the system level is required, and the versions must be consistent to ensure fast parallel compilation in JAX.
- JAX compilation depends on the soft link `/usr/local/cuda`; if it does not exist, create it manually:
  ```shell
  sudo ln -sf /usr/local/cuda-12.4 /usr/local/cuda
  ```
- Ensure that `nvcc -V` outputs correctly and that `which nvcc` points to the correct version.
- JAX Limitations: Only one `RWKV6_OP` object can be instantiated within the same program; the operator is thread-safe (stateless) and can be called from multiple places.
- JAX ≥ 0.6.0 no longer uses CUDA operators and defaults to native operators; version 0.4.34 is recommended.

### TensorFlow Usage Notes

- Only native API-based `RWKV6` operators are provided, which are only suitable for inference and have low efficiency.

---

### Usage

Note that unlike `rwkv7`, which is written as a function, `RWKV6` is a class that needs to be instantiated.
```python
from rwkv_ops import RWKV6_OP

operator = RWKV6_OP(
    head_size=64,               # Head size; use 64 if uncertain
    max_sequence_length=4096,   # Maximum training sequence length; inference is not limited
    ops_loop=False              # Optional: Whether to use the upper-level API instead of CUDA when sequence length = 1
)
```

#### Invocation

```python
y, y_state = operator(
    r, k, v, w, u,
    with_state=False,   # Whether to use a custom initial state / output final state
    init_state=None,    # Initial state [n_state, num_heads, head_size, head_size]
    state_map=None      # int32 one-dimensional array, length=batch_size, defining the init_state mapping
)
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| r, k, v, w | (batch_size, seq_len, hidden_size) | — |
| u | (num_heads, head_size) or (hidden_size,) | — |
| init_state | (n_state, num_heads, head_size, head_size) | When n_state=1, all samples share it; when n_state=batch_size, they correspond one-to-one |
| state_map | (batch_size,) | Specifies the init_state index for each sample |

| Return Value | Shape | Description |
|--------------|-------|-------------|
| y | (batch_size, seq_len, hidden_size) | Output |
| y_state | (batch_size, num_heads, head_size, head_size) or None | Final state |

---

### Distributed Tips

- The operator itself does not support distributed computing; PyTorch can directly use multi-threaded distributed computing.
- For JAX, use `shard_map` for packaging (example):

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
