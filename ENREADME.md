# RWKV OPS Project

> As RWKV continues to evolve, the core operators will be updated accordingly.  
> This repository is dedicated to maintaining the **operators** themselves, not layers or models; it aims to provide GPU-accelerated operators for various frameworks.

### Current Support
| Operator Type | Framework Support |
|---------------|-------------------|
| GPU operators | PyTorch, JAX      |
| Native operators | PyTorch, JAX, TensorFlow, NumPy |

> If the Keras ecosystem expands, MLX and OpenVINO may be supported in the future.  
> Note: This library depends on `keras`.

---

## Installation

```bash
pip install rwkv_ops
```

However, compiled operators are not fully removed by `pip uninstall`. You can install from source:

```bash
git clone https://github.com/pass-lin/rwkv_ops.git 
cd rwkv_ops
bash install.sh
```

---

## Environment Variables

| Variable Name | Meaning | Values | Default | Priority |
|---------------|---------|--------|---------|----------|
| `KERAS_BACKEND` | Keras backend | `jax` / `torch` / `tensorflow` / `numpy` | — | Low |
| `KERNEL_BACKEND` | Operator backend | `jax` / `torch` / `tensorflow` / `numpy` | `torch` | **High** |
| `KERNEL_TYPE` | Implementation type | `triton` / `cuda` / `native` | `cuda` | — |

> If `KERNEL_BACKEND` is set, it is used directly; if not, `KERAS_BACKEND` is used. If both are unset, `torch` is the default.

---

## Usage of `rwkv7op`

```python
from rwkv_ops import generalized_delta_rule  # or: from rwkv_ops import rwkv7_op, they are equivalent

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
        r:  [B, T, H, K]
        k:  [B, T, H, K]
        v:  [B, T, H, V]
        a:  [B, T, H, K]
        b:  [B, T, H, K]
        w:  [B, T, H, K]  # decay term in log space
        initial_state: initial state [N, H, K, V], N is number of sequences
        output_final_state: whether to return the final state
        head_first: whether to use head-first format (not supported for variable-length)

    Returns:
        o:           output [B, T, H, V] or [B, H, T, V]
        final_state: final state [N, H, K, V] or None
    """
```

### Special Notes for `torch-cuda`

- Under `torch-cuda`, `head_size` is also a kernel parameter, defaulting to 64.
- If `head_size ≠ 64`, use:

```python
from rwkv_ops import get_generalized_delta_rule

generalized_delta_rule, USE_TRITON_KERNEL = get_generalized_delta_rule(
    your_head_size, KERNEL_TYPE="cuda"
)
```

- `USE_TRITON_KERNEL` is a constant indicating whether the chunkwise kernel is used.
- Padding handling differs:

```python
if padding_mask is not None:
    w += (1 - padding_mask) * -1e9
```

- Loop-based kernels handle both left and right padding.
- For chunkwise kernels, **left padding is recommended**. CUDA and native kernels handle both correctly.

### Implementation Status of `rwkv7op`

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ✅     | ✅     |
| JAX         | ✅   | ✅     | ✅     |
| TensorFlow  | ⚠️    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| MLX         | ⚠️   | ❌     | ❌     |

1. `native`: no chunking, slow and memory-heavy.
2. `triton`: chunkwise implementation, fast and highly parallel, but **low precision**.
3. `cuda`: CUDA-based native operator, fast and high precision (fp32 internally), but may underutilize GPU for long sequences.
4. TensorFlow CUDA only supports forward pass (no gradients) and relies on JAX’s CUDA kernel.
5. MLX is not yet integrated into Keras, so native operators are not supported. A forward-only operator is provided.
6. tensorflow kernel only support eager mode
---

## Usage of `rwkv7_op_rnn`

### Background
This is a special case of RWKV7 OP for **sequence length = 1**, optimized for the **decoding stage** in inference.

### Usage

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
):
    """
    Single-step generalized delta rule (forward only).

    Args:
        r, w, k, v, a, b: input tensors, shape must be (B, 1, H, K) or (B, H, 1, K)
        initial_state: optional (B, H, K, K) initial state, zero-initialized if None
        output_final_state: whether to return the final state
        head_first: whether to move head dimension first

    Returns:
        out: (B, 1, H, K), same dtype as input
        last_state: (B, H, K, K) if output_final_state=True
    """
```

### Implementation Status of `rwkv7_op_rnn`

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ❌     | ✅     |
| JAX         | ✅   | ❌     | ✅     |
| TensorFlow  | ⚠️    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |

1. TensorFlow CUDA relies on JAX’s CUDA implementation.
2. Native implementation reuses `rwkv7_op`’s native code.
3. **This operator has no gradient support**.
4. tensorflow kernel only support eager mode
---

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
