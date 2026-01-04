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

## [MHC Operators](https://arxiv.org/abs/2512.24880)

Although unrelated to RWKV, these operators are integrated here for ease of distribution.

### Background

In multi-head architectures, the traditional approach usually involves simple linear transformations or weighting. MHC introduces the **Sinkhorn-Knopp** algorithm to constrain control weights within the space of doubly stochastic matrices, ensuring conservation and stability of information flow. Since these operations involve numerous intermediate variables and iterative calculations, native implementations are extremely memory-intensive. The CUDA operators provided in this library use **Operator Fusion** technology to significantly reduce memory consumption and improve execution speed.

Note: This repository provides a straightforward CUDA implementation, developed with reference to [mHC.cu](https://github.com/AndreSlavescu/mHC.cu) and assisted by Gemini. The current code successfully passes all numerical parity tests.

---

### MHC Operator List

| Operator Name | Core Functionality |
| --- | --- |
| `mhc_pre_op` | **Fused Pre-computation**: Handles input stream aggregation and Sinkhorn matrix preparation. |
| `mhc_post_op` | **Fused Post-computation**: Handles layer output distribution and multi-head residual fusion. |
| `sinkhorn_knopp` | Matrix doubly stochastic normalization (supports high-precision adjoint gradient). |
| `rmsnorm` | RMS normalization optimized for MHC input distributions. |
| `stream_aggregate` | Weighted feature stream aggregation (multi-stream to single-stream). |
| `stream_distribute` | Feature stream weighted distribution (single-stream to multi-stream). |
| `stream_mix` | Dynamic mixing between feature streams (Cross-head Mixing). |

---

### MHC Typical Integration Workflow

The MHC operator workflow is designed as a framework to replace traditional ResNet-style skip connections. A high-level usage example is shown below:

```python
from rwkv_ops import rmsnorm, mhc_pre_op, mhc_post_op

# 1. Normalize input
x_norm = rmsnorm(x_expanded)

# 2. Generate raw control parameters (typically via Linear layers)
# h_res_raw: [B, T, N, N], h_pre_raw/h_post_raw: [B, T, N]
h_res_raw, h_pre_raw, h_post_raw = linear_and_reshape(x_norm)

# 3. MHC Pre-computation (Fused Kernel)
x_layer_in, H_post, H_res = mhc_pre_op(
    x_expanded, h_pre_raw, h_post_raw, h_res_raw, num_iters=20
)

# 4. Execute core layer logic (x_layer_in is the aggregated single stream [B, T, C])
layer_out = YourCoreLayer(x_layer_in)

# 5. MHC Post-computation (Fused Kernel)
x_next = mhc_post_op(layer_out, x_expanded, H_post, H_res)

```

---

### Operator Definitions

#### 1. `mhc_pre_op` (Fused Pre-computation)

**Fusion Logic**: This operator is equivalent to fusing the following native operations:

* Applying `Sigmoid` to `h_pre_raw` to get pre-gating coefficients.
* Applying `Exp` + `Sinkhorn-Knopp` to `h_res_raw` to obtain the doubly stochastic weight matrix.
* Executing `stream_aggregate` on `x_expanded` (per-head aggregation).
* **Fusion Advantage**: Avoids storing massive exponential matrices and intermediate iteration states, reducing memory usage by approximately 80%.

**Interface**:

* **Inputs**:
* `x_expanded`: Expanded  feature heads.
* `h_pre_raw`: Raw pre-computation coefficients.
* `h_post_raw`: Raw post-computation coefficients.
* `h_res_raw`: Raw Sinkhorn inputs.


* **Returns**:
* `x_layer_in`: Fused layer input.
* `H_pre`, `H_post`, `H_res`: Normalized coefficients used by `post_op` and for backpropagation.



#### 2. `mhc_post_op` (Fused Post-computation)

**Fusion Logic**: This operator is equivalent to fusing the following native operations:

* Executing `stream_distribute` to map single-stream output back to multi-stream.
* Executing `stream_mix` using `H_res` for cross-head information exchange.
* Applying  to `H_post` as the residual gate.
* Executing multi-head residual addition: .

**Interface**:

* **Inputs**: `layer_out`, `x_expanded`, `H_post`, `H_res`.
* **Returns**: `x_next`.

#### 3. `sinkhorn_knopp`

* **Definition**: .
* **Features**: The CUDA implementation uses bidirectional iteration. Gradient calculation is implemented by solving the **Adjoint State Equation**, which is more numerically stable and memory-efficient than naive automatic differentiation through the iterative process.

#### 4. `rmsnorm`

* **Definition**: Standard RMS normalization. The CUDA implementation includes memory access optimizations specifically for the MHC feature distribution (typically optimized along the  or  dimensions). Following the mHC implementation, this version is parameter-less (no learnable affine transform).

#### 5. `stream_aggregate`

**Function**: Weighted spatial compression. It performs a linear weighted sum of  independent feature heads (streams) based on a given weight vector.

* **Mathematical Expression**: , where  and .
* **Equivalent Operation**: `torch.einsum('btn,btnc->btc', weights, x)`.

**Interface**:

* **Inputs**:
* `x`:  multi-stream feature tensor.
* `weights`:  weight coefficients for each stream.


* **Returns**:
* `out`:  aggregated single-stream features.



#### 6. `stream_distribute`

**Function**: Spatial broadcasting and re-weighting. It copies a single-stream feature into  channels and multiplies each by its corresponding distribution weight.

* **Mathematical Expression**: , where .
* **Equivalent Operation**: `x.unsqueeze(2) * weights.unsqueeze(-1)`.

**Interface**:

* **Inputs**:
* `x`:  single-stream features (usually from a core layer output).
* `weights`:  distribution weights for each head.


* **Returns**:
* `out`:  distributed multi-stream feature tensor.



#### 7. `stream_mix` (Stream Mixing)

**Function**: The key to cross-head communication in MHC. It uses an  transformation matrix (usually the Sinkhorn normalized matrix) to linearly recombine features across different heads.

* **Mathematical Expression**: .
* **Equivalent Operation**: `torch.einsum('btnm,btmc->btnc', gate, x)`.
* **Physical Significance**: Enables "non-local" information redistribution, allowing each head to absorb information from other heads.

**Interface**:

* **Inputs**:
* `x`:  raw multi-stream features.
* `gate`:  mixing matrix.


* **Returns**:
* `out`:  mixed multi-stream features.



---

### MHC Implementation Status

| Framework | cuda | triton | native |
| --- | --- | --- | --- |
| **PyTorch** | ✅ | ❌ | ✅ |
| **JAX** | ❌ | ❌ | ✅ |
| **TensorFlow** | ❌ | ❌ | ✅ |
| **NumPy** | ❌ | ❌ | ✅ |


---

## Environment Variables

| Variable Name | Meaning | Values | Default | Priority |
|---------------|---------|--------|---------|----------|
| `KERAS_BACKEND` | Keras backend | `jax` / `torch` / `tensorflow` / `numpy` | — | Low |
| `KERNEL_BACKEND` | Operator backend | `jax` / `torch` / `tensorflow` / `numpy` | `torch` | **High** |
| `KERNEL_TYPE` | Implementation type | `triton` / `cuda` / `native` | `cuda` | — |

> If `KERNEL_BACKEND` is set, it is used directly; if not, `KERAS_BACKEND` is used. If both are unset, `torch` is the default.

---

## rwkv7op Usage Guide

```python
from rwkv_ops import generalized_delta_rule, generalized_delta_rule_inference  # or: from rwkv_ops import rwkv7_op, they are identical
# generalized_delta_rule_inference has the same signature as generalized_delta_rule
# but it is inference-only (no gradients) and therefore saves some memory
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
        gk: [B, T, H, K]  # decay term in log-space!
        initial_state: initial state [N, H, K, V], N = number of sequences
        output_final_state: whether to return the final state
        head_first: whether to use head-first layout (variable length not supported)

    Returns:
        o:           output [B, T, H, V] or [B, H, T, V]
        final_state: final state [N, H, K, V] or None
    """
```

The only difference between `generalized_delta_rule_inference` and `generalized_delta_rule` is that the former does not compute gradients. Because activations need not be stored, memory consumption is reduced.

### CUDA-kernel special usage

- In the `torch-cuda` and `jax-cuda` kernels, `head_size` is also a kernel parameter; the default is 64.  
- If `head_size != 64`, use:

```python
from rwkv_ops import get_generalized_delta_rule

rwkv7_op, rwkv7_op_inference, USE_TRITON_KERNEL = get_generalized_delta_rule(
    your_head_size, KERNEL_TYPE="cuda"
)
```

- `USE_TRITON_KERNEL` is a constant that indicates whether the chunkwise kernel is being used.  
- The two kernels handle padding differently:

```python
if padding_mask is not None:
    w += (1 - padding_mask) * -1e9
```

- The loop-based kernel can cope with both left and right padding.  
- When the chunkwise kernel is used, **left padding is recommended**.  
  With the CUDA or native kernels, both left and right padding work correctly.

### rwkv7op implementation status

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ✅     | ✅     |
| JAX         | ✅   | ✅     | ✅     |
| TensorFlow  | ⚠️    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| MLX         | ⚠️   | ❌     | ❌     |

---

1. `native` = pure-Python / pure-JAX implementation, no chunking, slow and memory-hungry.  
2. `triton` = chunkwise Triton implementation, fast and highly parallel, but **numerical accuracy is poor—use only if you can tolerate the loss of precision**.  
3. `cuda` = hand-written CUDA kernel, very fast and internally uses FP32, so accuracy is high. Its weakness is throughput on very long sequences.  
4. TensorFlow’s CUDA support is forward-only (no gradients). It is actually a thin wrapper around JAX’s CUDA kernel; you must be able to run JAX’s CUDA kernel.  
5. The TensorFlow kernel works only in eager mode.  
6. MLX has not yet been merged into Keras, so the native kernel is currently unavailable. A forward-only operator is provided as a stop-gap.
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
