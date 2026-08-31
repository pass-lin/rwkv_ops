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
- [Usage of `rwkv7op_sane`](#usage-of-rwkv7op_sane)
  - [rwkv7op_sane implementation status](#rwkv7op_sane-implementation-status)
- [Usage of `rwkv7_op_sane_rnn`](#usage-of-rwkv7_op_sane_rnn)
  - [Implementation Status of `rwkv7_op_sane_rnn`](#implementation-status-of-rwkv7_op_sane_rnn)
- [Usage of `gdn_recurrent`](#usage-of-gdn_recurrent)
  - [API Reference](#api-reference-1)
    - [`gated_delta_net_recurrent`](#gated_delta_net_recurrent)
    - [`gated_delta_net_recurrent_inference`](#gated_delta_net_recurrent_inference)
    - [`gated_delta_net_recurrent_single_step`](#gated_delta_net_recurrent_single_step)
  - [Implementation Status of `gdn_recurrent`](#implementation-status-of-gdn_recurrent)
- [Usage of `gdn_recurrent_sane`](#usage-of-gdn_recurrent_sane)
  - [API Reference](#api-reference-2)
    - [`gated_delta_net_recurrent_sane`](#gated_delta_net_recurrent_sane)
    - [`gated_delta_net_recurrent_sane_inference`](#gated_delta_net_recurrent_sane_inference)
    - [`gated_delta_net_recurrent_sane_single_step`](#gated_delta_net_recurrent_sane_single_step)
  - [Implementation Status of `gdn_recurrent_sane`](#implementation-status-of-gdn_recurrent_sane)
- [Usage of `gdn_chunk`](#usage-of-gdn_chunk)
  - [API Reference](#api-reference-3)
    - [`gated_delta_net_chunk`](#gated_delta_net_chunk)
  - [Implementation Status of `gdn_chunk`](#implementation-status-of-gdn_chunk)
- [Usage of `gdn_chunk_sane`](#usage-of-gdn_chunk_sane)
  - [API Reference](#api-reference-4)
    - [`gated_delta_net_chunk_sane`](#gated_delta_net_chunk_sane)
  - [Implementation Status of `gdn_chunk_sane`](#implementation-status-of-gdn_chunk_sane)
- [Usage of `rwkv6op`](#usage-of-rwkv6op)
- [Factory functions and custom parameters](#factory-functions-and-custom-parameters)
- [Distributed Parallelism (JAX)](#distributed-parallelism)
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
| `RWKV_OPS_PALLAS_AUTOTUNE` | Pallas autotune switch | `1` / `0` | `0` | — |
| `RWKV_OPS_KERAS_NATIVE` | Force native to pure keras ops | `1` / `0` | `0` | — |

> With `KERNEL_TYPE=native`, the implementation is dispatched per backend and platform: jax + GPU/TPU uses the Pallas kernel (rwkv7/rwkv7_sane); torch + non-CPU uses the Triton kernel (rwkv7/rwkv7_sane/mhc, since pip torch bundles triton); everything else (CPU, mhc on jax, etc.) uses pure Keras ops. Set `RWKV_OPS_KERAS_NATIVE=1` to force pure Keras ops everywhere (for debugging).

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
    chunk_size: int = 16,
    mask=None,
):
    """
    Chunked Delta-Rule attention interface.

    Args:
        r:  [B, T, H, K]
        w:  [B, T, H, K]
        k:  [B, T, H, K]
        v:  [B, T, H, K]
        a:  [B, T, H, K]
        b:  [B, T, H, K]
        initial_state: initial state [B, H, K, K] or [1, H, K, K]
        output_final_state: whether to return the final state
        head_first: whether to use head-first layout (variable length not supported)
        chunk_size: chunk length, default 16; cuda/triton/pallas backends require T divisible by it
        mask: [B, T], decides whether the state is updated; 1 = update, 0 = freeze.
              Enabling mask roughly doubles training cost. Recommended alternative:
              `v *= mask; a *= mask; ops.where(mask, w, -1e9)`.

    Returns:
        o:           output [B, T, H, K] or [B, H, T, K]
        final_state: final state [B, H, K, K] or None
    """
```

The only difference between `generalized_delta_rule_inference` and `generalized_delta_rule` is that the former does not compute gradients. Because activations need not be stored, memory consumption is reduced.

<a id="cuda-kernel-special-usage"></a>
### CUDA-kernel special usage

- In the `torch-cuda` and `jax-cuda` kernels, both `head_size` and `chunk_size` are compile-time constants; the defaults are 64 and 16, respectively.  
- If you need `head_size != 64` or `chunk_size != 16`, use the factory function:

```python
from rwkv_ops import get_generalized_delta_rule

rwkv7_op, rwkv7_op_inference = get_generalized_delta_rule(
    HEAD_SIZE=your_head_size, KERNEL_TYPE="cuda", chunk_size=32
)
```

- `chunk_size` is passed into the CUDA kernel via the `-D_CHUNK_LEN_` macro, so a different `chunk_size` triggers a separate compilation and produces an independent `.so` / torch extension. Switching `chunk_size` will not accidentally reuse the old kernel.
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

> ¹ With the JAX backend, `native` on GPU/TPU is the Pallas implementation (`jax_pallas_kernel.py`); with the Torch backend, `native` on non-CPU platforms is the Triton implementation; elsewhere it is pure Keras ops.


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

<a id="usage-of-rwkv7op_sane"></a>
## Usage of `rwkv7op_sane`

`rwkv7op_sane` implements **State Anomaly Neutralization (SANE)**, a numerical stabilization technique that softly clips the RWKV-7 state at chunk boundaries. See the paper [SANE: State Anomaly Neutralization for RWKV-7](https://arxiv.org/pdf/2608.22354) for details.

```python
from rwkv_ops import generalized_delta_rule_sane, generalized_delta_rule_sane_inference

def generalized_delta_rule_sane(
    r,
    w,
    k,
    v,
    a,
    b,
    tau,                  # [B, T//chunk_size, H], float32, already softplus(param)+1, must be > 0
    mask=None,            # [B, T//chunk_size], float32, 1 -> apply SANE, 0 -> skip; shared across heads
    initial_state=None,
    output_final_state: bool = True,
    head_first: bool = False,
    chunk_size: int = 16,
):
    """
    RWKV-7 generalized delta rule with State Anomaly Neutralization (training / prefill).

    Dispatch rules:
    - When ``output_final_state=False``, the internal no-mask operator is used,
      applying State Anomaly Neutralization unconditionally at every chunk boundary to save the
      mask read/branch overhead.
    - When ``mask=None`` and ``output_final_state=True``, the no-mask operator is
      also used, but a warning is raised and the returned ``final_state`` is set
      to ``None`` to prevent users from accidentally using a state that may be
      contaminated by padding chunks.
    - The masked operator is used only when ``output_final_state=True`` and an
      explicit ``mask`` is provided.

    Args:
        r, w, k, v, a, b: [B, T, H, K] or [B, H, T, K], T must be divisible by chunk_size.
        tau: [B, T//chunk_size, H], float32, must be > 0.
        mask: [B, T//chunk_size], float32, 0/1 per-chunk flag for State Anomaly Neutralization;
              only effective when output_final_state=True and mask is explicitly
              provided. Set padded chunks to 0 and keep k=0, a=0, w=-inf.
        initial_state: [B, H, K, K] or [1, H, K, K].
        output_final_state: whether to return the final state.
        head_first: whether input is head-first.
        chunk_size: chunk length, default 16. For the CUDA backend it is a compile-time constant and must be set via the factory function; for Triton/Pallas/native it can be passed at call time.

    Returns:
        out: [B, T, H, K]
        final_state: [B, H, K, K] or None (when mask=None and output_final_state=True)
    """
```

`generalized_delta_rule_sane_inference` has the same interface but **does not compute gradients**, saving memory.
Note: the inference kernel reads `tau` per chunk, so `tau` only needs to have length `T // chunk_size`; **T is no longer required to be divisible by chunk_size**. For arbitrary-length prefill, you can also use the single-step RNN interface below.

<a id="rwkv7op_sane-implementation-status"></a>
### rwkv7op_sane implementation status

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ✅     | ✅     |
| JAX         | ✅   | ✅     | ✅¹    |
| TensorFlow  | ❌    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

---

<a id="usage-of-rwkv7_op_sane_rnn"></a>
## Usage of `rwkv7_op_sane_rnn`

```python
from rwkv_ops import rwkv7_op_sane_rnn

def rwkv7_op_sane_rnn(
    r,                    # [B, 1, H, K] or [B, H, 1, K]
    w,
    k,
    v,
    a,
    b,
    tau,                  # [B, H], float32
    do_sane,                # bool or [B] bool, True -> apply State Anomaly Neutralization after this step
    initial_state=None,
    output_final_state: bool = True,
    head_first: bool = False,
):
    """
    RWKV-7 single-step inference with State Anomaly Neutralization (RNN mode).
    Output is computed from the pre-SANE state; state_out has SANE applied when do_sane is True.
    """
```

Example (trigger SANE every chunk_size steps, default 16):

```python
for step in range(seq_len):
    do_sane = (step % chunk_size == chunk_size - 1)
    out, state = rwkv7_op_sane_rnn(
        r[step], w[step], k[step], v[step], a[step], b[step],
        tau=tau, do_sane=do_sane, initial_state=state
    )
```

<a id="implementation-status-of-rwkv7_op_sane_rnn"></a>
### Implementation Status of `rwkv7_op_sane_rnn`

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ✅   | ❌     | ✅     |
| JAX         | ✅   | ❌     | ✅     |
| TensorFlow  | ❌    | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

1. Single-step operator **has no gradient support**.
2. The CUDA version casts inputs to bfloat16 internally, same as `rwkv7_op_rnn`.

<a id="usage-of-gdn_recurrent"></a>
## Usage of `gdn_recurrent`

`gdn_recurrent` provides a step-by-step recurrent implementation of **Gated DeltaNet**, with training, inference, and single-step RNN entry points. The default input layout is `[B, T, H, K/V]`; set `head_first=True` to use `[B, H, T, K/V]`.

```python
from rwkv_ops import (
    gated_delta_net_recurrent,
    gated_delta_net_recurrent_inference,
    gated_delta_net_recurrent_single_step,
)

# Training / prefill (gradient-capable)
out, final_state = gated_delta_net_recurrent(
    q, k, v, g, beta,
    initial_state=h0,
    output_final_state=True,
    head_first=False,
)

# Inference-only (no gradients, lower memory)
out, final_state = gated_delta_net_recurrent_inference(
    q, k, v, g, beta,
    initial_state=h0,
    output_final_state=True,
    head_first=False,
)

# Single-step RNN (decode stage)
out, state = gated_delta_net_recurrent_single_step(
    q, k, v, g, beta,
    initial_state=state,
    output_final_state=True,
    head_first=True,  # single-step currently requires head_first=True
)
```

<a id="api-reference-1"></a>
### API Reference

<a id="gated_delta_net_recurrent"></a>
#### `gated_delta_net_recurrent`

| Parameter | Shape | Description |
|---|---|---|
| q, k | (B, T, H, K) | Query and key; internally L2-normalized |
| v | (B, T, H, V) | Value |
| g | (B, T, H) | Log-space decay gate |
| beta | (B, T, H) | Write strength; must already be sigmoid-activated to (0, 1) |
| initial_state | (B, H, K, V) or (1, H, K, V), optional | Initial recurrent state |
| output_final_state | bool | Whether to return the final state |
| head_first | bool | Whether inputs/outputs use head-first layout |
| chunk_size | int | Chunk length, default 16; ignored by the plain recurrent implementation (kept for signature consistency) |

| Return value | Shape | Description |
|---|---|---|
| out | (B, T, H, V) | Same dtype as `v` |
| final_state | (B, H, K, V) or None | Final state |

<a id="gated_delta_net_recurrent_inference"></a>
#### `gated_delta_net_recurrent_inference`

Same interface as `gated_delta_net_recurrent`, but **does not compute gradients** and therefore avoids storing `kv_mem`, `state_chkp`, and other reverse-only intermediates.

<a id="gated_delta_net_recurrent_single_step"></a>
#### `gated_delta_net_recurrent_single_step`

| Parameter | Shape | Description |
|---|---|---|
| q, k | (B, H, K) | Single-step query and key |
| v | (B, H, V) | Single-step value |
| g | (B, H) | Single-step log-space decay gate |
| beta | (B, H) | Single-step write strength, already sigmoid-activated |
| initial_state | (B, H, K, V) or (1, H, K, V), optional | Current state |
| output_final_state | bool | Whether to return the next state |
| head_first | bool | Single-step currently only supports `head_first=True` |
| chunk_size | int | Chunk length, default 16; ignored by the single-step implementation (kept for signature consistency) |

| Return value | Shape | Description |
|---|---|---|
| out | (B, H, V) | Same dtype as `v` |
| next_state | (B, H, K, V) | State for the next step |

<a id="implementation-status-of-gdn_recurrent"></a>
### Implementation Status of `gdn_recurrent`

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ❌   | ✅     | ✅     |
| JAX         | ❌   | ✅     | ✅¹    |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

> ¹ With the JAX backend, `native` on GPU/TPU uses the Pallas implementation (`jax_pallas_kernel.py`); `triton` is the JAX-Triton implementation when `KERNEL_TYPE="triton"` is set explicitly; elsewhere it falls back to pure Keras ops.

1. The training entry point supports back-propagation; the inference and single-step entry points **do not support gradients**.
2. The chunkwise counterpart lives in `gdn_chunk/`, see below.

<a id="usage-of-gdn_recurrent_sane"></a>
## Usage of `gdn_recurrent_sane`

`gdn_recurrent_sane` extends `gdn_recurrent` with **State Anomaly Neutralization (SANE)**: at every chunk boundary the state is soft-clipped via `state = tau * tanh(state / tau)`, which helps prevent padding chunks or anomalous states from contaminating the recurrent state. It also exposes training, inference, and single-step RNN entry points.

```python
from rwkv_ops import (
    gated_delta_net_recurrent_sane,
    gated_delta_net_recurrent_sane_inference,
    gated_delta_net_recurrent_sane_single_step,
)

# Training / prefill (gradient-capable)
out, final_state = gated_delta_net_recurrent_sane(
    q, k, v, g, beta, tau, mask=mask,
    initial_state=h0,
    output_final_state=True,
    head_first=False,
)

# Inference-only (no gradients, lower memory)
out, final_state = gated_delta_net_recurrent_sane_inference(
    q, k, v, g, beta, tau, mask=mask,
    initial_state=h0,
    output_final_state=True,
    head_first=False,
)

# Single-step RNN (decode stage)
out, state = gated_delta_net_recurrent_sane_single_step(
    q, k, v, g, beta, tau, do_sane,
    initial_state=state,
    output_final_state=True,
    head_first=True,  # single-step currently only supports head_first=True
)
```

<a id="api-reference-2"></a>
### API Reference

<a id="gated_delta_net_recurrent_sane"></a>
#### `gated_delta_net_recurrent_sane`

| Parameter | Shape | Description |
|---|---|---|
| q, k | (B, T, H, K) | Query and key; L2-normalized inside the operator |
| v | (B, T, H, V) | Value |
| g | (B, T, H) | Log-space decay gate |
| beta | (B, T, H) | Write strength; must have already passed through sigmoid, i.e. in (0, 1) |
| tau | (B, T//chunk_size, H) | SANE threshold; must be > 0 |
| mask | (B, T//chunk_size), optional | SANE is applied at chunk boundaries where mask > 0; only effective when `output_final_state=True` |
| initial_state | (B, H, K, V) or (1, H, K, V), optional | Initial recurrent state |
| output_final_state | bool | Whether to return the final state |
| head_first | bool | Whether the input/output is head-first |
| chunk_size | int | SANE chunk length, default 16; determines the chunk dimension of `tau`/`mask` |

| Return value | Shape | Description |
|---|---|---|
| out | (B, T, H, V) | Same dtype as `v`; computed from the state **before** SANE |
| final_state | (B, H, K, V) or None | Final state; `None` with a `UserWarning` when `mask=None` and `output_final_state=True` |

<a id="gated_delta_net_recurrent_sane_inference"></a>
#### `gated_delta_net_recurrent_sane_inference`

Same interface as `gated_delta_net_recurrent_sane`, but **does not compute gradients** and therefore avoids storing `kv_mem`, `state_chkp`, and other reverse-only intermediates. Supports arbitrary sequence length `T`.

<a id="gated_delta_net_recurrent_sane_single_step"></a>
#### `gated_delta_net_recurrent_sane_single_step`

| Parameter | Shape | Description |
|---|---|---|
| q, k | (B, H, K) | Single-step query and key |
| v | (B, H, V) | Single-step value |
| g | (B, H) | Single-step log-space decay gate |
| beta | (B, H) | Single-step write strength, already sigmoided |
| tau | (B, H) | Single-step SANE threshold |
| do_sane | (B,) | Apply SANE where > 0, skip otherwise |
| initial_state | (B, H, K, V) or (1, H, K, V), optional | Current state |
| output_final_state | bool | Whether to return the next state |
| head_first | bool | Single-step currently only supports `head_first=True` |
| chunk_size | int | Chunk length, default 16; ignored by the single-step implementation (kept for signature consistency) |

| Return value | Shape | Description |
|---|---|---|
| out | (B, H, V) | Same dtype as `v` |
| next_state | (B, H, K, V) | State for the next step |

<a id="implementation-status-of-gdn_recurrent_sane"></a>
### Implementation Status of `gdn_recurrent_sane`

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ❌   | ✅     | ✅     |
| JAX         | ❌   | ✅     | ✅¹    |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

> ¹ With the JAX backend, `native` on GPU/TPU uses the Pallas implementation (`jax_pallas_kernel.py`); `triton` is the JAX-Triton implementation when `KERNEL_TYPE="triton"` is set explicitly; elsewhere it falls back to pure Keras ops.

1. The training entry point supports back-propagation (including `tau` gradients); the inference and single-step entry points **do not support gradients**.
2. When `mask=None`, unconditional SANE is still performed, but `output_final_state=True` emits a `UserWarning` and sets `final_state=None` to avoid using a state potentially contaminated by padding.

<a id="usage-of-gdn_chunk"></a>
## Usage of `gdn_chunk`

`gdn_chunk` provides a **chunkwise parallel** training implementation of Gated DeltaNet. The default input layout is `[B, T, H, K/V]`; internally it is transposed to `[B, H, T, K/V]` to match the Triton kernels.

```python
from rwkv_ops import gated_delta_net_chunk

out, final_state = gated_delta_net_chunk(
    q, k, v, g, beta,
    initial_state=h0,
    output_final_state=True,
    chunk_size=16,
)
```

<a id="api-reference-3"></a>
### API Reference

<a id="gated_delta_net_chunk"></a>
#### `gated_delta_net_chunk`

| Argument | Shape | Description |
|---|---|---|
| q, k | (B, T, H, K) | Query and key; L2-normalized inside the operator |
| v | (B, T, H, V) | Value |
| g | (B, T, H) | Log-space decay gate |
| beta | (B, T, H) | Write strength; must have already passed through sigmoid, i.e. in (0, 1) |
| initial_state | (B, H, K, V) or (1, H, K, V), optional | Initial recurrent state |
| output_final_state | bool | Whether to return the final state |
| chunk_size | int | Chunk length; must divide T |

| Return | Shape | Description |
|---|---|---|
| out | (B, T, H, V) | Same dtype as `v` |
| final_state | (B, H, K, V) or None | Final state |

<a id="implementation-status-of-gdn_chunk"></a>
### Implementation Status of `gdn_chunk`

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ❌   | ✅     | ✅     |
| JAX         | ❌   | ✅     | ✅     |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

> The training entry point supports back-propagation. On the JAX side, `triton` requires explicit `KERNEL_TYPE="triton"` and the `jax-triton` package.

---

<a id="usage-of-gdn_chunk_sane"></a>
## Usage of `gdn_chunk_sane`

`gdn_chunk_sane` extends `gdn_chunk` with **State Anomaly Neutralization (SANE)**: at every chunk boundary the cross-chunk state is transformed by `state = tau * tanh(state / tau)`, gated by `mask`. Other behaviors match `gdn_chunk`.

```python
from rwkv_ops import gated_delta_net_chunk_sane

out, final_state = gated_delta_net_chunk_sane(
    q, k, v, g, beta, tau,
    mask=mask,                 # [B, T//chunk_size], optional
    initial_state=h0,
    output_final_state=True,
    chunk_size=16,
)
```

<a id="api-reference-4"></a>
### API Reference

<a id="gated_delta_net_chunk_sane"></a>
#### `gated_delta_net_chunk_sane`

| Argument | Shape | Description |
|---|---|---|
| q, k | (B, T, H, K) | Query and key; L2-normalized inside the operator |
| v | (B, T, H, V) | Value |
| g | (B, T, H) | Log-space decay gate |
| beta | (B, T, H) | Write strength; must have already passed through sigmoid, i.e. in (0, 1) |
| tau | (B, T//chunk_size, H) | SANE threshold; must be > 1 |
| mask | (B, T//chunk_size), optional | SANE is applied where > 0; unconditional SANE when None |
| initial_state | (B, H, K, V) or (1, H, K, V), optional | Initial recurrent state |
| output_final_state | bool | Whether to return the final state; when mask is None final_state is forced to None with a UserWarning |
| chunk_size | int | Chunk length; must divide T |

| Return | Shape | Description |
|---|---|---|
| out | (B, T, H, V) | Same dtype as `v` |
| final_state | (B, H, K, V) or None | Final state; None when mask is None |

<a id="implementation-status-of-gdn_chunk_sane"></a>
### Implementation Status of `gdn_chunk_sane`

| Framework   | cuda | triton | native |
|-------------|------|--------|--------|
| PyTorch     | ❌   | ✅     | ✅     |
| JAX         | ❌   | ✅     | ✅     |
| TensorFlow  | ❌   | ❌     | ✅     |
| NumPy       | ❌   | ❌     | ✅     |
| OpenVINO    | ❌   | ❌     | ✅     |

> The training entry point supports back-propagation (including gradients for `tau`). On the JAX side, `triton` requires explicit `KERNEL_TYPE="triton"` and the `jax-triton` package.
> When `mask=None` the output still uses unconditional SANE, but `output_final_state=True` emits a `UserWarning` and sets `final_state` to `None`.

---

<a id="distributed-parallelism"></a>
## Distributed Parallelism (JAX)

All accelerated JAX operators declare sharding propagation rules via
`custom_partitioning` + einsum-style `sharding_rule`, so they partition
correctly under `jax.jit` + `NamedSharding`, supporting **DP (data parallel
over batch)** and **TP (tensor parallel over heads)**:

| Operator (jax) | DP (batch) | TP (head) |
|---|---|---|
| rwkv7 cuda / triton / pallas | ✅ | ✅ |
| rwkv7_sane cuda / triton / pallas | ✅ | ✅ |
| rwkv7 / rwkv7_sane single-step cuda | ✅ | ✅ |
| gdn_recurrent triton / pallas | ✅ | ✅ |
| gdn_recurrent_sane triton / pallas | ✅ | ✅ |
| gdn_chunk triton | ✅ | ✅ |
| gdn_chunk_sane triton | ✅ | ✅ |
| rwkv6 cuda | ✅ | ❌ |

> rwkv6 fuses the channel dim (C = H × N) into a single rule dimension, so the
> head axis is not exposed to the partitioner; only batch parallelism is
> supported. Use rwkv7 / rwkv7_sane if you need TP.

**Only shard the batch or head dims**; sharding time or head_size dims yields
wrong results (the scan needs the full T, and each state needs the full
head_size). SANE's `tau` carries a head dim (TP-shardable), while `mask` has no
head dim (replicated automatically under TP).

Example (TP over heads):

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

For DP, shard the batch dim onto the mesh axis (the rules also cover the
backward pass and the sharding propagation of state checkpoints / final_state).
Inference-side DP is usually process-level replication and does not go through
a mesh; the single-step TP rules target TP inference deployments.

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


<a id="factory-functions-and-custom-parameters"></a>
## Factory functions and custom parameters

`from rwkv_ops import *` instantiates a set of **default operators** with `HEAD_SIZE=64`, `chunk_size=16`, and `KERNEL_TYPE` taken from the `KERNEL_TYPE` environment variable (default `cuda`).

To customize `head_size`, `chunk_size`, or `KERNEL_TYPE`, use the corresponding `get_*` factory function:

```python
from rwkv_ops import (
    get_generalized_delta_rule,           # RWKV-7
    get_generalized_delta_rule_sane,      # RWKV-7-SANE
    get_gated_delta_net_chunk,            # GDN chunkwise
    get_gated_delta_net_chunk_sane,       # GDN chunkwise SANE
    get_gated_delta_net_recurrent,        # GDN recurrent (non-SANE)
    get_gated_delta_net_recurrent_sane,   # GDN recurrent SANE
)

# RWKV-7 CUDA: HEAD_SIZE and chunk_size are compile-time constants and must be set in the factory
rwkv7_op, rwkv7_op_inference = get_generalized_delta_rule(
    HEAD_SIZE=64, KERNEL_TYPE="cuda", chunk_size=32
)

# RWKV-7-SANE CUDA: chunk_size must also be set in the factory
rwkv7_op_sane, rwkv7_op_sane_inference = get_generalized_delta_rule_sane(
    HEAD_SIZE=64, KERNEL_TYPE="cuda", chunk_size=32
)

# Gated DeltaNet chunkwise (chunk_size can be passed at call time for triton/native)
gdn_chunk = get_gated_delta_net_chunk(KERNEL_TYPE="triton", chunk_size=32)

# Gated DeltaNet chunkwise SANE (chunk_size can be passed at call time for triton/native)
gdn_chunk_sane = get_gated_delta_net_chunk_sane(
    KERNEL_TYPE="triton", chunk_size=32
)

# Gated DeltaNet recurrent (non-SANE; chunk_size is ignored, kept only for signature consistency)
gdn_recurrent = get_gated_delta_net_recurrent(KERNEL_TYPE="triton", chunk_size=32)

# Gated DeltaNet recurrent SANE (chunk_size is used for SANE boundaries)
gdn_recurrent_sane = get_gated_delta_net_recurrent_sane(
    KERNEL_TYPE="triton", chunk_size=32
)
```

**Notes:**
- **CUDA backend**: `chunk_size` is baked into the kernel via the `-D_CHUNK_LEN_` macro, so it must be specified through the factory function. The returned operator still has a `chunk_size` argument, but passing a different value at call time will raise an error. Each distinct `chunk_size` triggers a separate compilation. Affected operators: `generalized_delta_rule`, `generalized_delta_rule_sane`, `gated_delta_net_chunk`, `gated_delta_net_chunk_sane`, `gated_delta_net_recurrent_sane`.
- **Triton / Pallas / native backends**: `chunk_size` can be passed at call time; the factory only sets the default. Changing `chunk_size` does not trigger recompilation.
- `chunk_size` defaults to 16; `gated_delta_net_recurrent` (non-SANE) accepts but ignores this parameter, keeping it only for signature consistency.
- Factory functions return operators with the same signature as the default operators and can be called directly.


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
