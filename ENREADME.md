# RWKV OPS Project

> As RWKV continues to evolve, the core operators will be updated accordingly.  
> This repository is dedicated to maintaining the **operators** themselves, not layers or models; it aims to provide GPU-accelerated operators for various frameworks.

<a id="current-support"></a>
### Current Support
| Operator Type | Framework Support |
|---------------|-------------------|
| GPU operators | PyTorch, JAX      |
| Native operators | PyTorch, JAX, TensorFlow, NumPy |

> If the Keras ecosystem expands, MLX and OpenVINO may be supported in the future.  

> Note: This library depends on `keras`.

## Table of Contents

  - [Current Support](#current-support)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [mHC Operations Usage](#mhc-operations-usage)
  - [Quick Start](#quick-start)
  - [API Reference](#api-reference)
    - [`mhc_pre_op`](#mhc_pre_op)
    - [`mhc_post_op`](#mhc_post_op)
  - [mHC Implementation Status](#mhc-implementation-status)
- [rwkv7op Usage Guide](#rwkv7op-usage-guide)
  - [CUDA-kernel special usage](#cuda-kernel-special-usage)
  - [rwkv7op implementation status](#rwkv7op-implementation-status)
- [Usage of `rwkv7_op_rnn`](#usage-of-rwkv7_op_rnn)
  - [Background](#background)
  - [Usage](#usage)
  - [Implementation Status of `rwkv7_op_rnn`](#implementation-status-of-rwkv7_op_rnn)
- [Usage of `rwkv7op_sn`](#usage-of-rwkv7op_sn)
  - [rwkv7op_sn implementation status](#rwkv7op_sn-implementation-status)
- [Usage of `rwkv7_op_sn_rnn`](#usage-of-rwkv7_op_sn_rnn)
  - [Implementation Status of `rwkv7_op_sn_rnn`](#implementation-status-of-rwkv7_op_sn_rnn)
- [Usage of `rwkv6op`](#usage-of-rwkv6op)
  - [PyTorch Usage Notes](#pytorch-usage-notes)
  - [JAX Usage Notes](#jax-usage-notes)
  - [TensorFlow Usage Notes](#tensorflow-usage-notes)
  - [Usage](#usage-1)
  - [rwkv6op Implementation Status](#rwkv6op-implementation-status)
- [Testing](#testing)

---

<a id="installation"></a>
## Installation

```bash
pip install rwkv_ops
```


---

<a id="environment-variables"></a>
## Environment Variables

| Variable Name | Meaning | Values | Default | Priority |
|---------------|---------|--------|---------|----------|
| `KERAS_BACKEND` | Keras backend | `jax` / `torch` / `tensorflow` / `numpy` / `openvino` | — | Low |
| `KERNEL_BACKEND` | Operator backend | `jax` / `torch` / `tensorflow` / `numpy` / `openvino` | `torch` | **High** |
| `KERNEL_TYPE` | Implementation type | `triton` / `cuda` / `native` | `cuda` | — |
| `RWKV_OPS_PALLAS_AUTOTUNE` | Pallas autotune switch | `1` / `0` | `1` | — |
| `RWKV_OPS_JAX_NATIVE` | jax native impl choice | `pallas` (default) / `xla` | — | — |

> With the JAX backend, when `KERNEL_TYPE=native` and the platform is GPU/TPU, the `native` implementation of rwkv7/rwkv7_sn is the Pallas kernel (pure Keras ops on other platforms, including CPU). The Pallas kernels only use public Pallas APIs, and autotune picks the fastest compilable backend candidate.

> If `KERNEL_BACKEND` is set, it is used directly; if not, `KERAS_BACKEND` is used. If both are unset, `torch` is the default.

---

---

<a id="mhc-operations-usage"></a>
## mHC Operations Usage

[mHC (Multi-Head Control)](https://arxiv.org/pdf/2512.24880) is a new residual interaction mechanism introduced by DeepSeek as an evolution/replacement for standard ResNet. It extends the traditional single-stream residual connection into a parallel multi-stream architecture, introducing dynamic aggregation and distribution.

This repository provides Keras-compatible kernels implemented in **Triton**. As this is a practice project for the author to learn Triton, performance optimization has not reached the theoretical limit:
* **JAX Backend**: XLA's fusion capabilities are formidable, making the Triton speedup less significant (Native latency is ~1.5x ResNet, Triton is ~1.27x, while DeepSeek's expert-optimized version is ~1.06x). However, in terms of **VRAM usage**, the Triton operator uses a custom VJP to force recomputation, saving **3~4GB VRAM** for a full model (tested at `128x1024x4x768`) compared to JAX Native.
* **Torch Backend**: Since `torch.compile` is currently less efficient at fusing such complex logic than XLA, the Triton operator shows significant advantages (**Pre-Op is ~8x faster, Post-Op is ~3x faster** in single-op benchmarks). **It is highly recommended for Torch users to enable Triton by default.** (Note: These benchmarks refer to individual operators).

<a id="quick-start"></a>
### Quick Start

```python
from rwkv_ops import mhc_pre_op, mhc_post_op

# Alternatively, explicitly get the kernel (default is "triton")
# mhc_pre_op, mhc_post_op = get_mhc_kernel("triton")

# Example usage within a layer (e.g., Attention/FFN):
# 1. Pre-Op: Aggregate multi-stream into a single stream
x_layer_in, h_post, h_res = mhc_pre_op(
    x, alpha_pre, alpha_post, alpha_res, phi, 
    bias_pre, bias_post, bias_res, n=4
)

# 2. Core Layer Computation
x_layer_out = attention(x_layer_in)

# 3. Post-Op: Distribute back to multi-stream and perform stream mixing
x_next = mhc_post_op(x_layer_out, x, h_post, h_res)
```

---

<a id="api-reference"></a>
### API Reference

<a id="mhc_pre_op"></a>
#### `mhc_pre_op`
Aggregates multi-stream features into the core layer input and generates coefficients for subsequent stages.

| Parameter | Shape | Description |
|---|---|---|
| x | (B, T, n, C) | Multi-stream input features |
| alpha_pre/post/res | (1,) | Scaling scalars for Aggregate, Distribute, and Residual branches |
| phi | (n*C, n*(n+2)) | Dynamic projection matrix |
| bias_pre/post/res | (M,) | Bias terms for each branch |
| n | int | Expansion rate (number of streams) |
| num_iters | int | Sinkhorn-Knopp iteration count (default: 20) |

| Return Value | Shape | Description |
|---|---|---|
| x_layer_in | (B, T, C) | Aggregated single-stream feature for Attention/FFN |
| h_post_raw | (B, T, n) | Distribution weights (unactivated) for Post-Op |
| H_res | (B, T, n, n) | Doubly stochastic residual mixing matrix for Post-Op |

---

<a id="mhc_post_op"></a>
#### `mhc_post_op`
Distributes the core layer output back to multiple streams using gated weights and updates the stream states via the mixing matrix.

| Parameter | Shape | Description |
|---|---|---|
| layer_out | (B, T, C) | Output from the core layer (Attention/FFN) |
| x_expanded | (B, T, n, C) | Multi-stream state before Pre-Op (residual path data) |
| h_post_raw | (B, T, n) | Distribution weights from Pre-Op |
| H_res | (B, T, n, n) | Residual mixing matrix from Pre-Op |

| Return Value | Shape | Description |
|---|---|---|
| x_next | (B, T, n, C) | Updated multi-stream features for the next layer |

---
**Constraint: The channel dimension `C` must be divisible by 128.**

<a id="mhc-implementation-status"></a>
### mHC Implementation Status

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ❌   | ✅     | ✅      |
| JAX       | ❌   | ✅     | ✅      |
| TensorFlow| ❌   | ❌     | ✅      |
| NumPy     | ❌   | ❌     | ✅      |
| OpenVINO  | ❌   | ❌     | ✅      |

> **Implementation Notes:**
> 1. **Recommended for Torch**: On A100, `mhc_post_op` is ~3x faster than `torch.compile`, and `mhc_pre_op` is ~8x faster.
> 2. **Choice for JAX Users**: XLA's native performance is strong. For pure inference throughput, using Native `mhc_pre_op` paired with Triton `mhc_post_op` is suggested. While Triton `mhc_pre_op` is ~2x faster and `mhc_post_op` is ~1.1x faster in single-op tests, XLA can perform deeper fusion across the entire graph. However, the **Triton version is significantly more VRAM-efficient**, making it the preferred choice for training BERT-like or GPT-like deep models.
> 3. **Consistency**: JAX and Torch share the same Triton logic; performance differences arise from how each backend schedules external kernels (XLA has superior graph-packing, while Torch is currently weaker in this regard).

---
<a id="rwkv7op-usage-guide"></a>
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
    mask=None,
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
        mask [B,T] decide state update
        initial_state: initial state [N, H, K, V], N = number of sequences
        output_final_state: whether to return the final state
        head_first: whether to use head-first layout (variable length not supported)

    Returns:
        o:           output [B, T, H, V] or [B, H, T, V]
        final_state: final state [N, H, K, V] or None
    """
```

The only difference between `generalized_delta_rule_inference` and `generalized_delta_rule` is that the former does not compute gradients. Because activations need not be stored, memory consumption is reduced.

<a id="cuda-kernel-special-usage"></a>
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

<a id="rwkv7op-implementation-status"></a>
### rwkv7op implementation status

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ❌     | ✅     |
| JAX         | ✅   | ❌     | ✅¹    |
| TensorFlow  | ❌    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

> ¹ With the JAX backend, `native` on GPU/TPU is the Pallas implementation (`jax_pallas_kernel.py`); on other platforms it is pure Keras ops.


---

1. `native` = pure-Python / pure-JAX implementation, no chunking, slow and memory-hungry.  
2. `cuda` = hand-written CUDA kernel, very fast and internally uses FP32, so accuracy is high. Its weakness is throughput on very long sequences.  

---

<a id="usage-of-rwkv7_op_rnn"></a>
## Usage of `rwkv7_op_rnn`

<a id="background"></a>
### Background
This is a special case of RWKV7 OP for **sequence length = 1**, optimized for the **decoding stage** in inference.

<a id="usage"></a>
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

<a id="implementation-status-of-rwkv7_op_rnn"></a>
### Implementation Status of `rwkv7_op_rnn`

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ❌     | ✅     |
| JAX         | ✅   | ❌     | ✅     |
| TensorFlow  | ❌    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

1. Native implementation reuses `rwkv7_op`’s native code.
2. **This operator has no gradient support**.

---

<a id="usage-of-rwkv7op_sn"></a>
## Usage of `rwkv7op_sn`

```python
from rwkv_ops import generalized_delta_rule_sn, generalized_delta_rule_sn_inference

def generalized_delta_rule_sn(
    r,
    w,
    k,
    v,
    a,
    b,
    tau,                  # [B, T//16, H], float32, already softplus(param)+1, must be > 0
    mask=None,            # [B, T//16], float32, 1 -> apply SN, 0 -> skip; shared across heads
    initial_state=None,
    output_final_state: bool = True,
    head_first: bool = False,
):
    """
    RWKV-7 generalized delta rule with State Neutralization (training / prefill).

    Dispatch rules:
    - When ``output_final_state=False``, the internal no-mask operator is used,
      applying State Neutralization unconditionally at every chunk boundary to save the
      mask read/branch overhead.
    - When ``mask=None`` and ``output_final_state=True``, the no-mask operator is
      also used, but a warning is raised and the returned ``final_state`` is set
      to ``None`` to prevent users from accidentally using a state that may be
      contaminated by padding chunks.
    - The masked operator is used only when ``output_final_state=True`` and an
      explicit ``mask`` is provided.

    Args:
        r, w, k, v, a, b: [B, T, H, K] or [B, H, T, K], T must be divisible by 16.
        tau: [B, T//16, H], float32, must be > 0.
        mask: [B, T//16], float32, 0/1 per-chunk flag for State Neutralization;
              only effective when output_final_state=True and mask is explicitly
              provided. Set padded chunks to 0 and keep k=0, a=0, w=-inf.
        initial_state: [B, H, K, K] or [1, H, K, K].
        output_final_state: whether to return the final state.
        head_first: whether input is head-first.

    Returns:
        out: [B, T, H, K]
        final_state: [B, H, K, K] or None (when mask=None and output_final_state=True)
    """
```

`generalized_delta_rule_sn_inference` has the same interface but **does not compute gradients**, saving memory.
Note: the inference kernel reads `tau` per chunk, so `tau` only needs to have length `T // 16`; **T is no longer required to be divisible by 16**. For arbitrary-length prefill, you can also use the single-step RNN interface below.

<a id="rwkv7op_sn-implementation-status"></a>
### rwkv7op_sn implementation status

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ✅     | ✅     |
| JAX         | ✅   | ✅     | ✅¹    |
| TensorFlow  | ❌    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

---

<a id="usage-of-rwkv7_op_sn_rnn"></a>
## Usage of `rwkv7_op_sn_rnn`

```python
from rwkv_ops import rwkv7_op_sn_rnn

def rwkv7_op_sn_rnn(
    r,                    # [B, 1, H, K] or [B, H, 1, K]
    w,
    k,
    v,
    a,
    b,
    tau,                  # [B, H], float32
    do_sn,                # bool or [B] bool, True -> apply State Neutralization after this step
    initial_state=None,
    output_final_state: bool = True,
    head_first: bool = False,
):
    """
    RWKV-7 single-step inference with State Neutralization (RNN mode).
    Output is computed from the pre-SN state; state_out has SN applied when do_sn is True.
    """
```

Example (trigger SN every 16 steps):

```python
for step in range(seq_len):
    do_sn = (step % 16 == 15)
    out, state = rwkv7_op_sn_rnn(
        r[step], w[step], k[step], v[step], a[step], b[step],
        tau=tau, do_sn=do_sn, initial_state=state
    )
```

<a id="implementation-status-of-rwkv7_op_sn_rnn"></a>
### Implementation Status of `rwkv7_op_sn_rnn`

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ❌     | ✅     |
| JAX         | ✅   | ❌     | ✅     |
| TensorFlow  | ❌    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

1. Single-step operator **has no gradient support**.
2. The CUDA version casts inputs to bfloat16 internally, same as `rwkv7_op_rnn`.

---

<a id="usage-of-rwkv6op"></a>
## Usage of `rwkv6op`

<a id="pytorch-usage-notes"></a>
### PyTorch Usage Notes

- Dependencies: `keras`, `ninja`, and a complete CUDA toolkit.
- If using VS Code with a virtual environment for debugging, make sure to manually activate the virtual environment in the terminal before running the code; otherwise, ninja may not work.
- Although PyTorch can run normally even if the CUDA version in the virtual environment is inconsistent with the global CUDA version, it is strongly recommended to keep them consistent.
- The operator is thread-safe (stateless) and can be called from multiple places.

<a id="jax-usage-notes"></a>
### JAX Usage Notes

- Dependencies: `keras`, `cmake`, `gcc`, and a complete CUDA toolkit.
- Even if CUDA is installed for JAX via a virtual environment, a complete CUDA installation at the system level is required, and the versions must be consistent to ensure fast parallel compilation in JAX.
- JAX compilation depends on the soft link `/usr/local/cuda`; if it does not exist, create it manually:
  ```shell
  sudo ln -sf /usr/local/cuda-12.4 /usr/local/cuda
  ```
- Ensure that `nvcc -V` outputs correctly and that `which nvcc` points to the correct version.
- JAX `cuda` backend uses `jax.ffi` and supports JAX >= 0.4.31 (including 0.6.x). The CUDA path only accelerates `bfloat16`; non-`bfloat16` inputs trigger a warning and are cast to `bfloat16`, no longer falling back to `native`.

<a id="tensorflow-usage-notes"></a>
### TensorFlow Usage Notes

- Only native API-based `RWKV6` operators are provided, which have low efficiency.

---

<a id="usage-1"></a>
### Usage

Like `rwkv7`, `RWKV6` now exposes a **functional interface**.

```python
from rwkv_ops import rwkv6_op  # or backward-compatible alias RWKV6_OP

y, final_state = rwkv6_op(
    r, k, v, w, u,
    initial_state=None,
    output_final_state=False,
    state_map=None,
    head_first=False,
)
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| r, k, v, w | (B, T, C) or (B, H, T, N) | — |
| u | (H, N) or (C,) | — |
| initial_state | (S, H, N, N) or (H, N, N) | S=1 shared; S=B one-to-one |
| state_map | (B,) int64 | Index into initial_state for each sample |

| Return Value | Shape | Description |
|--------------|-------|-------------|
| y | (B, T, C) or (B, H, T, N) | Output |
| final_state | (B, H, N, N) or None | Final state |

> For a custom `head_size` or `max_sequence_length`, use:
> ```python
> from rwkv_ops import get_rwkv6_kernel
> rwkv6_op = get_rwkv6_kernel(HEAD_SIZE=64, KERNEL_TYPE="cuda", MAX_SEQUENCE_LENGTH=4096)
> ```
> `MAX_SEQUENCE_LENGTH` is a compile-time constant `_T_` for the CUDA kernel and must be no less than the actual sequence length; the `native` implementation ignores this parameter.




---

<a id="rwkv6op-implementation-status"></a>
### rwkv6op Implementation Status

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ❌     | ✅     |
| JAX         | ✅   | ❌     | ✅     |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

JAX `cuda` backend is based on `jax.ffi` and supports JAX >= 0.4.31 (including 0.6.x). The CUDA path only accelerates `bfloat16`; non-`bfloat16` inputs trigger a warning and are cast to `bfloat16`, no longer falling back to `native`.


<a id="testing"></a>
## Testing

The project has been migrated to pytest, with tests isolated by backend:

```bash
# Install test dependencies
pip install -e ".[test]"

# torch backend (covers rwkv6/rwkv7 CUDA, inference, single-step, triton, mHC)
pytest tests/torch -v

# jax backend (requires cmake and a compatible GCC)
pytest tests/jax -v

# numpy / tensorflow only run native smoke tests
pytest tests/numpy -v
pytest tests/tensorflow -v

# Skip heavier slow tests
pytest tests/torch tests/jax -v -m "not slow"
```

After each pytest session, `build_*` directories, `.so` files, and `__pycache__` are automatically cleaned up.

> **Note**: Test files in different backend directories use unique module names (e.g. `test_torch_rwkv6.py` / `test_jax_rwkv6.py`) so that collecting from the `tests` root does not hit pytest's `import file mismatch`. Keep filenames unique when adding cross-backend tests.
>
> **JAX CUDA compilation compatibility**: `rwkv_ops` automatically uses the in-tree `rwkv_ops/cuda_tools/nvcc_wrap` to work around the `rsqrt` header conflict between glibc 2.41+ and CUDA 13.1, so no system CUDA header patching is required.
