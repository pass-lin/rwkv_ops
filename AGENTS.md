# RWKV OPS 项目 —— 给 AI 协作代理的说明

> 本文件面向自动编程 Agent，补充 README 中未详细描述的工程约定、代码结构与协作规范。
> 项目维护的是 RWKV 系列模型的**核心算子**，不维护 layer 或完整 model。
> 项目依赖 `keras>=3.0`，通过 `KERAS_BACKEND` 统一支持 PyTorch、JAX、TensorFlow、NumPy 多个后端。

---

## 1. 项目定位与总体结构

```text
rwkv_ops/
├── __init__.py              # 包入口：读取环境变量、实例化全部算子、暴露公共 API
├── pallas_utils.py          # Pallas 后端公共机制（候选配置 / autotune / SPMD 辅助）
├── cuda_tools/              # nvcc_wrap + 绕过 CUDA/glibc 冲突的头文件
├── gdn_chunk/               # Gated DeltaNet chunkwise 算子
├── gdn_recurrent/           # Gated DeltaNet recurrent / reference 算子
├── mhc_kernel/              # mHC (Multi-Head Control) 算子
├── rwkv6_kernel/            # RWKV-6 算子
├── rwkv7_kernel/            # RWKV-7 广义 delta rule 算子
└── rwkv7_sane_kernel/         # RWKV-7 State Anomaly Neutralization 算子

tests/                       # pytest 测试目录（按后端隔离）
├── conftest.py              # 公共 fixtures / 数值对比工具（不 import 任何后端！）
├── jax/                     # JAX 后端测试
├── numpy/                   # NumPy 后端测试
├── tensorflow/              # TensorFlow 后端测试
└── torch/                   # PyTorch 后端测试

clean_build_artifacts.py     # 编译产物清理（pytest 会话结束自动调用）
pyproject.toml               # hatchling 构建配置
MANIFEST.in                  # 源码分发清单
```

五个算子家族相互独立，但共享同一套**后端选择机制**和**原生 Keras 参考实现**。

### 1.1 暴露的公共 API

在 `rwkv_ops/__init__.py` 中统一导出（import 即按环境变量实例化）：

| 名称 | 说明 |
|---|---|
| `generalized_delta_rule` / `rwkv7_op` | RWKV-7 训练算子（chunkwise） |
| `generalized_delta_rule_inference` / `rwkv7_op_inference` | RWKV-7 推理算子（无梯度，省显存） |
| `rnn_generalized_delta_rule` / `rwkv7_op_rnn` | RWKV-7 单步算子（T=1，用于 decode） |
| `generalized_delta_rule_sane` / `rwkv7_op_sane` | RWKV-7-SANE 训练算子 |
| `generalized_delta_rule_sane_inference` / `rwkv7_op_sane_inference` | RWKV-7-SANE 推理算子（T 不必被 16 整除） |
| `rnn_generalized_delta_rule_sane` / `rwkv7_op_sane_rnn` | RWKV-7-SANE 单步算子 |
| `rwkv6_op` / `RWKV6_OP` | RWKV-6 函数式算子（`RWKV6_OP` 为兼容别名） |
| `gated_delta_net_recurrent` / `gated_delta_net_recurrent_inference` / `gated_delta_net_recurrent_single_step` | Gated DeltaNet recurrent 算子 |
| `gated_delta_net_recurrent_sane` / `gated_delta_net_recurrent_sane_inference` / `gated_delta_net_recurrent_sane_single_step` | Gated DeltaNet recurrent SANE 算子 |
| `gated_delta_net_chunk` | Gated DeltaNet chunkwise 算子 |
| `gated_delta_net_chunk_sane` | Gated DeltaNet chunkwise SANE 算子 |
| `delta_net_recurrent` / `delta_net_recurrent_inference` / `delta_net_recurrent_single_step` | DeltaNet recurrent 算子（训练/推理/单步） |
| `delta_net_recurrent_sane` / `delta_net_recurrent_sane_inference` / `delta_net_recurrent_sane_single_step` | DeltaNet recurrent SANE 算子（训练/推理/单步） |
| `delta_net_chunk` | DeltaNet chunkwise 算子 |
| `delta_net_chunk_sane` | DeltaNet chunkwise SANE 算子 |
| `mhc_pre_op` / `mhc_post_op` | mHC 预处理/后处理算子 |
| `get_generalized_delta_rule` 等 22 个工厂函数 | 按 head_size / KERNEL_TYPE / chunk_size 获取算子 |

---

## 2. 后端选择机制（非常重要）

### 2.1 环境变量

| 变量 | 含义 | 可取值 | 默认值 | 优先级 |
|---|---|---|---|---|
| `KERNEL_BACKEND` | 算子后端 | `jax` / `torch` / `tensorflow` / `numpy` / `openvino` | `torch` | **最高** |
| `KERAS_BACKEND` | Keras 后端 | `jax` / `torch` / `tensorflow` / `numpy` / `openvino` | — | 低 |
| `KERNEL_TYPE` | 实现类型 | `triton` / `cuda` / `native` | `cuda` | — |
| `RWKV_OPS_PALLAS_AUTOTUNE` | Pallas autotune 开关 | `1` / `0` | `0` | — |
| `RWKV_OPS_KERAS_NATIVE` | 强制 native 为纯 keras ops | `1` / `0` | `0` | — |
| `RWKV_OPS_PALLAS_BACKEND` | Pallas 后端强制覆盖 | `default` / `mgpu` / `triton` | — | — |

选择逻辑（见 `rwkv_ops/__init__.py`）：

1. 若 `KERNEL_BACKEND` 有值，直接使用；
2. 否则若 `KERAS_BACKEND` 有值，使用它；
3. 否则默认 `torch`，并自动 `keras.config.set_backend("torch")`。

`KERNEL_TYPE` 决定具体实现：

- `native`：可移植实现，"装好库即可用"。具体实现按后端与平台分发：
  - **jax + GPU/TPU** = Pallas kernel（rwkv7/rwkv7_sane；pallas 随 jax 自带）。
  - **torch + 非 CPU（CUDA/ROCm/XPU）** = Triton kernel（rwkv7/rwkv7_sane/mhc；
    pip 版 torch 自带 triton；XPU/ROCm 未实测，按"非 CPU 且 triton 可导入"开放）。
  - 其余（CPU、tensorflow、numpy、openvino、mhc 的 jax 侧）= 纯 Keras ops
    （ground truth）。mhc 的 jax triton 桥接依赖额外的 jax-triton 包，
    不属于"装好库即可用"，必须显式 `KERNEL_TYPE="triton"`。
  - 设 `RWKV_OPS_KERAS_NATIVE=1` 可强制 jax/torch 的 native 都用纯 keras ops
    （无 kernel 的调试/数值对照）。
- `cuda`：手写 CUDA kernel（Torch C++ 扩展 / JAX FFI）。
- `triton`：Triton 实现（rwkv7 / rwkv7_sane / gdn_chunk / gdn_chunk_sane / gdn_recurrent / gdn_recurrent_sane / mhc）。

**缺硬件静默回退**：各工厂在硬件/库不可用时不报错，直接回退 native
（例如 torch 无 CUDA、jax 不在 GPU/TPU 上时回到纯 keras ops）。

### 2.1.x chunk_size 的指定方式

`chunk_size` 不是环境变量，而是算子/工厂函数的参数，默认 16。不同后端对其处理不同：

- **CUDA 后端**：`chunk_size` 是编译期常量（`-D_CHUNK_LEN_`），必须通过工厂函数指定；返回的算子仍带 `chunk_size` 参数，但传入与编译值不同的 `chunk_size` 会报错。不同 `chunk_size` 会各自编译一次，互不影响。涉及算子：`generalized_delta_rule`、`generalized_delta_rule_sane`、`gated_delta_net_chunk`、`gated_delta_net_chunk_sane`、`gated_delta_net_recurrent_sane`、`delta_net_recurrent_sane`。
- **Triton / Pallas / native 后端**：`chunk_size` 在每次调用时作为 `tl.constexpr` 或直接参数传入，无需工厂指定，可在调用时修改。

`gated_delta_net_recurrent`（无 SANE）的 native/Triton 实现接受 `chunk_size` 但忽略，
仅保持签名一致；其 PyTorch CUDA 实现把 `chunk_size` 作为编译期常量按
`(K, V, chunk_size)` 在首次调用时懒编译，调用时传入不同值会各自编译一次，
不经过工厂指定也不做一致性校验。

### 2.2 各算子的后端支持矩阵

#### RWKV-7 `generalized_delta_rule`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ✅   | ✅     | ✅     |
| JAX       | ✅   | ✅     | ✅¹    |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

#### RWKV-7-SANE `generalized_delta_rule_sane`

同 rwkv7op（PyTorch/JAX 全后端 ✅，其余仅 native）。

> ¹ JAX 后端的 `native` 在 GPU/TPU 上为 Pallas 实现（`jax_pallas_kernel.py`），
> Torch 后端的 `native` 在非 CPU 平台为 Triton 实现，其余为纯 Keras ops；
> 单步 RNN 无 pallas 版本。

#### RWKV-7 `rwkv7_op_rnn` / SANE `rwkv7_op_sane_rnn` (T=1)

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ✅   | ❌     | ✅     |
| JAX       | ✅   | ❌     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

#### RWKV-6 `rwkv6_op`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ✅   | ❌     | ✅     |
| JAX       | ✅   | ❌     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

> JAX `cuda` 后端通过 `jax.ffi` 实现，需要 JAX >= 0.4.31。
> RWKV-6 的 `KERNEL_TYPE="triton"` 未实现，**静默回退 native**。
> ROCm (HIP) 仅 Torch 路径保留（`RWKV_USE_ROCM=1`）；JAX 路径不支持。

#### mHC `mhc_pre_op` / `mhc_post_op`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ❌   | ✅     | ✅²    |
| JAX       | ❌   | ✅     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

> ² Torch 后端的 `native` 在非 CPU 平台默认为 Triton 实现；jax 侧 `native`
> 保持纯 Keras ops（jax triton 桥接依赖额外的 jax-triton 包）。

#### Gated DeltaNet chunkwise `gated_delta_net_chunk`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ❌   | ✅     | ✅     |
| JAX       | ❌   | ✅     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

> PyTorch / JAX 后端的 `triton` 已实现 chunkwise 训练 kernel（前向 + 反向）。
> JAX 侧需显式 `KERNEL_TYPE="triton"` 且安装 `jax-triton`。

#### Gated DeltaNet chunkwise SANE `gated_delta_net_chunk_sane`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ❌   | ✅     | ✅     |
| JAX       | ❌   | ✅     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

> PyTorch / JAX 后端的 `triton` 已实现 chunkwise SANE 训练 kernel（前向 + 反向，
> 含 `tau` 梯度）。JAX 侧需显式 `KERNEL_TYPE="triton"` 且安装 `jax-triton`。
> `mask=None` 时仍执行无条件 SANE，但 `output_final_state=True` 会发出 `UserWarning`
> 并将 `final_state` 置为 `None`。

#### Gated DeltaNet recurrent `gated_delta_net_recurrent` / `gated_delta_net_recurrent_inference` / `gated_delta_net_recurrent_single_step`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ✅   | ✅     | ✅     |
| JAX       | ✅   | ✅     | ✅³    |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

> PyTorch 侧 `gdn_recurrent` 已提供 Triton 前向 kernel（训练/推理/单步 RNN 三个入口）
> 与 CUDA kernel（训练含反向、推理、单步 RNN 三个入口，`torch_cuda_kernel/`）；
> CUDA 版 `chunk_size` 作为编译期常量按 `(K, V, chunk_size)` 在首次调用时懒编译，
> 调用时传入不同 `chunk_size` 会各自编译一次（不做工厂值校验）；
> JAX 侧 `cuda` 为 FFI 实现（`jax_cuda_kernel/`，训练含反向、推理、单步 RNN 三个入口，
> bf16/fp32 双实例化，同样按 `(K, V, chunk_size)` 懒编译）；
> JAX 侧 `triton` 显式 `KERNEL_TYPE="triton"` 时启用 JAX-Triton 前向 kernel，
> `native` 在 GPU/TPU 上为 Pallas 实现（`jax_pallas_kernel.py`），并覆盖训练/推理/单步
> RNN 三个入口。

> ³ JAX 后端的 `native` 在 GPU/TPU 上为 Pallas 实现，其余为纯 Keras ops；
> `triton` 需要显式 `KERNEL_TYPE="triton"` 且安装 `jax-triton`。

#### Gated DeltaNet recurrent SANE `gated_delta_net_recurrent_sane` / `gated_delta_net_recurrent_sane_inference` / `gated_delta_net_recurrent_sane_single_step`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ✅   | ✅     | ✅     |
| JAX       | ✅   | ✅     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

> PyTorch 侧 `gdn_recurrent_sane` 已提供 Triton kernel（训练/推理/单步 RNN 三个入口）
> 与 CUDA kernel（训练含反向含 `tau` 梯度、推理、单步 RNN 三个入口，`torch_cuda_kernel/`）；
> CUDA 版 `chunk_size` 作为编译期常量按 `(K, V, chunk_size)` 在首次调用时懒编译，
> 调用时传入不同 `chunk_size` 会各自编译一次（不做工厂值校验）；
> JAX 侧 `cuda` 为 FFI 实现（`jax_cuda_kernel/`，训练含反向、推理、单步 RNN 三个入口，
> bf16/fp32 双实例化，同样按 `(K, V, chunk_size)` 懒编译）；
> JAX 侧 `triton` 显式 `KERNEL_TYPE="triton"` 时启用 JAX-Triton 前向 kernel，
> `native` 在 GPU/TPU 上为 Pallas 实现（`jax_pallas_kernel.py`），覆盖训练/推理/单步
> RNN 三个入口；其余环境回退 native Keras ops。
> `mask=None` 时仍执行无条件 SANE，但 `output_final_state=True` 会发出 `UserWarning` 并将
> `final_state` 置为 `None`（与 `rwkv7_sane` 语义一致）。

#### DeltaNet recurrent `delta_net_recurrent` / `delta_net_recurrent_inference` / `delta_net_recurrent_single_step`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ✅   | ✅     | ✅²    |
| JAX       | ✅   | ✅     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

> ² Torch 后端的 `native` 在非 CPU 平台默认为 Triton 实现；JAX 侧 `triton`
> 需显式 `KERNEL_TYPE="triton"` 且安装 `jax-triton`，`native` 在 GPU/TPU 上
> 为 Pallas 实现（`jax_pallas_kernel.py`），其余为纯 Keras ops。
> PyTorch 侧 `cuda` 为 C++/CUDA 扩展（`torch_cuda_kernel/`，训练含反向、
> 推理、单步三个入口）；JAX 侧 `cuda` 为 FFI 实现（`jax_cuda_kernel/`，
> bf16/fp32 双实例化，训练含反向、推理、单步三个入口，需 JAX >= 0.4.31）；
> CUDA 版 `chunk_size` 作为编译期常量按 `(K, V, chunk_size)` 懒编译，
> 调用时传入不同值会各自编译一次（不做工厂值校验）。
> 训练入口含反向，推理与单步入口仅前向。

#### DeltaNet chunkwise `delta_net_chunk`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ❌   | ✅     | ✅³    |
| JAX       | ❌   | ✅     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

> ³ Torch 后端的 `native` 在非 CPU 平台默认为 Triton 实现；JAX 侧 `triton`
> 需显式 `KERNEL_TYPE="triton"` 且安装 `jax-triton`，`native` 为纯 Keras ops。
> chunk 家族只做 native + Triton，训练入口含反向。
> recurrent 家族的 CUDA/Pallas 加速内核在后续阶段提供（见 §14）。

#### DeltaNet recurrent SANE `delta_net_recurrent_sane` / `delta_net_recurrent_sane_inference` / `delta_net_recurrent_sane_single_step`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ✅   | ✅     | ✅     |
| JAX       | ✅   | ✅     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

> Torch 后端的 `native` 在非 CPU 平台默认为 Triton 实现；JAX 侧 `triton`
> 需显式 `KERNEL_TYPE="triton"` 且安装 `jax-triton`，`native` 在 GPU/TPU 上为 Pallas 实现。
> `cuda` 在 PyTorch（C++ 扩展）与 JAX（FFI，需 JAX >= 0.4.31）均提供训练（含反向含 `tau` 梯度）
> /推理/单步三入口，`chunk_size` 作为编译期常量按 `(K, V, chunk_size)` 懒编译。
> 训练反向的 `dtau` 由 `atomic_add` 累加，Triton autotune `pre_hook` / JAX `zeroed_outputs`
> 保证其初值为零。

#### DeltaNet chunkwise SANE `delta_net_chunk_sane`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ❌   | ✅     | ✅     |
| JAX       | ❌   | ✅     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

> Torch 后端的 `native` 在非 CPU 平台默认为 Triton 实现；JAX 侧 `triton`
> 需显式 `KERNEL_TYPE="triton"` 且安装 `jax-triton`，`native` 为纯 Keras ops。
> chunk 家族只做 native + Triton，训练入口含反向（含 tau 梯度）。
> Triton 反向的 `dtau` 由 `atomic_add` 累加，autotune benchmark 会重复执行 kernel，
> 因此其 autotune 注册了 `pre_hook` 在每次 benchmark 前清零 `dtau`。

### 2.3 分布式分片（jax）

jax 侧所有加速算子都用 `custom_partitioning` + einsum 风格 `sharding_rule`
声明分片传播，支持 **DP（batch 维并行）** 与 **TP（head 维并行）**：

| 算子（jax） | DP (batch) | TP (head) |
|---|---|---|
| rwkv7 cuda / triton / pallas | ✅ | ✅ |
| rwkv7_sane cuda / triton / pallas | ✅ | ✅（含 1-device TP 结构测试） |
| rwkv7 / rwkv7_sane 单步 cuda | ✅ | ✅ |
| gdn_recurrent cuda / triton / pallas | ✅ | ✅ |
| gdn_recurrent_sane cuda / triton / pallas | ✅ | ✅ |
| delta_net_recurrent cuda / triton / pallas | ✅ | ✅ |
| delta_net_recurrent_sane cuda / triton / pallas | ✅ | ✅ |
| gdn_chunk_sane triton | ✅ | ✅ |
| delta_net_chunk_sane triton | ✅ | ✅ |
| rwkv6 cuda | ✅ | ❌（channel 融合为 `c`，head 维未暴露） |

- 规则字母：`b`=batch、`n`/`h`=head、`t`=time、`k`/`m`/`n`=head_size、
  `c`=chunk（rwkv6 的 `c` 是融合 channel）。
- **只允许切 batch 或 head 维**；切 time 或 head_size 维会静默算错
  （扫描需要完整 T，state 需要完整 head_size）。
- SANE 的 `tau`/`mask` 已按规则覆盖：tau 带 head 维（TP 可切）、mask 无 head 维
  （TP 下自动 replicate）；单步的 `do_sane` 仅 batch 维。
- `infer_sharding_from_operands` 按输出实际维度重建 `NamedSharding`
  （state checkpoint / final_state / dtau 等不是输入同形张量）。
- 单卡可用 1-device mesh 做结构验证（编译通过 + 输出 spec 正确传播 +
  数值一致），真正的多卡行为需在多卡环境复核。
- 推理侧的 DP 通常是进程级副本（不经 mesh），所以 DP 规则主要服务于
  单进程 SPMD 训练；单步算子的 TP 规则服务于 TP 推理部署。

---

## 3. 代码组织约定

### 3.1 每个算子家族的目录结构模式

以 `rwkv7_kernel` 为例（`rwkv7_sane_kernel` 完全平行）：

```text
rwkv7_kernel/
├── __init__.py                    # 工厂函数，根据后端+KERNEL_TYPE 选择实现
├── native_keras_op.py             # 纯 Keras 参考实现（ground truth）
├── triton_kernel.py               # 共享 Triton 内核（被 JAX/Torch 桥接共用）
├── jax_triton_kernel.py           # JAX ↔ Triton 桥接（custom_vjp + SPMD）
├── jax_pallas_kernel.py           # JAX Pallas 内核（GPU/TPU，native/pallas 时默认）
├── torch_triton_kernel.py         # PyTorch ↔ Triton 桥接（autograd.Function）
├── jax_cuda_kernel/               # JAX FFI CUDA（wkv7_jax.py + wkv7_ffi.cu + CMakeLists.txt）
├── jax_cuda_kernel_single/        # JAX FFI CUDA 单步
├── torch_cuda_kernel/             # PyTorch C++/CUDA 扩展
└── torch_cuda_kernel_single/      # PyTorch C++/CUDA 单步
```

`gdn_chunk/` 与 `gdn_recurrent/` 采用同样的目录约定，但拆成两个家族：
chunkwise 版本专门放分块并行实现（训练 / 推理），recurrent 版本放逐步
参考实现与单步 decode 实现。加速内核按 `KERNEL_TYPE` 在各自家族目录下扩展
（如 `gdn_recurrent/torch_cuda_kernel/`）。

### 3.2 原生实现的地位

- `native_keras_op.py` / `ops_rwkv_kernel.py` / `native_op.py` 是**数值对齐的基准**。
- 新增/修改任何加速实现时，必须保证与原生实现的数值一致性。
- 原生实现使用 `keras.ops` 编写，因此天然跨后端。

### 3.3 精度约定

- 原生实现内部大量使用 `float32` 计算，最后 `cast` 回输入 dtype。
- CUDA/Triton/Pallas 内核通常 `bfloat16` I/O、`float32` 内部累加；
  State/tau/中间量保持 fp32。dtype 不符时**警告并 cast，不报错**。
- RWKV-6 的 fp16 输入会输出 fp32；fp32/bf16 保持同类型。
- mHC Triton 内核的 `H_post` / `H_res` 输出固定为 `float32`。

### 3.4 共享与复用

- **共享 Triton kernel**：改 `triton_kernel.py` 必须同步检查 jax/torch 两个桥接。
- **JAX SPMD 三件套**：`custom_partitioning` + `def_partition(
  infer_sharding_from_operands, sharding_rule, partition)` + `_create_partition`
  样板。sharding rule 用同一字母标记所有 head 维度以支持 head 轴 TP。
- **Pallas 公共机制**：全部集中在 `rwkv_ops/pallas_utils.py`，新增 pallas
  kernel 必须复用，不要内联复制。

---

## 4. 关键实现细节

### 4.1 RWKV-7 广义 delta rule

核心递推（见 `native_keras_op.py`）：

```text
w_t = exp(-exp(w_t))                          # decay
sa_t = state_{t-1} @ a_t
state_t = state_{t-1} * w_t + sa_t ⊗ b_t + v_t ⊗ k_t
y_t = state_t @ r_t
```

- 状态 `state` 形状：`(B, H, K, K)`，`H` 是 head 数，`K` 是 head_size。
- 输入默认 layout `[B, T, H, K]`；`head_first=True` 时内部转置为 `[B, H, T, K]`。
- `mask` 为 **per-token** `[B, T]`（或 `[B, T, 1, 1]`），1 更新状态、0 冻结状态。
- chunkwise 内核要求 `T % chunk_size == 0`（cuda 训练版 / triton / pallas），`chunk_size` 默认 16，可通过工厂函数或算子参数修改。
- `get_generalized_delta_rule` 返回 `(训练算子, 推理算子)`；推理版不存 checkpoint。
- `triton` 后端固定支持 `HEAD_SIZE == 64`；其他 head_size 用 `cuda`
  （`head_size` 经 `-D_C_` 编译进内核，按 head_size 懒编译）。
- 单步 `get_rnn_generalized_delta_rule` 只支持 cuda，其余 KERNEL_TYPE 回退 native。
- 单步 cuda 桥接（rwkv7/sane）带 `custom_partitioning` 分片规则，支持 DP（batch）
  与 TP（head）：`b h k ... b h k m -> b h k, b h k m`；SANE 单步的 tau 为
  `[B, H]`（per-head）、`do_sane` 为 `[B]`（per-sample），规则同样覆盖。

### 4.2 RWKV-7 State Anomaly Neutralization（`rwkv7_sane_kernel`）

SANE（State Anomaly Neutralization）是一种在 chunk 边界对 RWKV-7 的 state 做软裁剪的数值稳定技术，详见论文 [SANE: State Anomaly Neutralization for RWKV-7](https://arxiv.org/pdf/2608.22354)。

在 RWKV-7 递推之上，于 chunk 边界（每 chunk_size 个 token）做 State Anomaly Neutralization：

```text
sane_state = tau * tanh(state / tau)      # 软裁剪到 [-tau, tau]
```

- **chunk_size 语义**：SANE chunk 长度，默认 16。CUDA 后端作为编译期常量需通过工厂函数指定；Triton/Pallas/native 后端可在算子调用时直接传入。
- **tau 语义**：只接收预处理后的 `tau = softplus(param) + 1.0`，严格 > 1，
  形状 `[B, T//chunk_size, H]`（per-head per-chunk）。
- **mask 语义**：`[B, T//chunk_size]` per-chunk，所有 head 共享。mask=1 在 chunk 边界
  执行 SANE；mask=0 保留原 state。不再用 `tau=0.0` 兼任 mask；全 padding chunk
  的 mask 须置 0。kernel 用 `mask * sane_state + (1 - mask) * state` 的 blend 形式，
  避免 warp 分支。
- **padding 处理**：padding 位仍需保证 `k=0, a=0, w=-inf`，且对应 chunk mask=0。
- **输出与 State 的关系**：输出始终基于 SANE **之前**的 State；SANE 只修改传递给
  下一步/下一 chunk 的 State。训练 kernel 的 checkpoint 保存 **SANE 之前** 的
  State 供反向使用；反向先算 `dtau`，再把 `dstate`/`dstateT` 乘 `sech2`。
- **无 mask 算子**：`output_final_state=False` 或 `mask=None` 时调用独立
  no-mask kernel（chunk 边界无条件 SANE，不读 mask、不算 blend）。
  `mask=None` 且 `output_final_state=True` 时 Python 入口发双语 UserWarning
  并把 `final_state` 置为 `None`（避免误用被 padding 污染的 state）。
- **推理 kernel**：`generalized_delta_rule_sane_inference` 只输出 y 与最终 state，
  显存显著低于训练版；按 chunk 读 tau，**T 不要求被 chunk_size 整除**。任意长度
  prefill 也可用单步 `generalized_delta_rule_sane_single_step`（每步算 SANE，
  按 per-sample `do_sane` 选择）。
- **分片**：JAX `custom_partitioning` 的 sharding rule 支持 head 轴 TP；
  `infer_sharding_from_operands` 按输出实际维度重建 `NamedSharding`；
  mask 无 head 维，自动 replicate；Torch 侧按 head 独立 launch 即可 TP。

### 4.3 RWKV-6

- `ops_rwkv_kernel.py` 的 `RWKVKernelOperator` 是数值 ground truth（while_loop
  逐步 RNN，State 全程 fp32，`w = exp(-exp(w))`）；`native_keras_op.py` 是函数式
  薄封装，每次调用 new 一个 operator。
- 函数式接口：`rwkv6_op(r, k, v, w, u, initial_state=None,
  output_final_state=False, state_map=None, head_first=False)`。
- `max_sequence_length` 是 CUDA kernel 的**编译期常量**，不同
  `(head_size, max_sequence_length)` 组合各自懒编译一个 `.so`；运行时
  `T > MAX_SEQUENCE_LENGTH` 会显式 raise。
- 输入 layout `[B, T, C]`（或 head_first 时 `[B, H, T, N]`），`C = H * N`；
  `u` 形状 `(H, N)` 或 `(C,)`。
- `initial_state` 可为 `[H,N,N]` 或 `[B,H,N,N]`；batch 维为 1 或 B 时自动推断
  `state_map`，否则必须显式传 `[B]` int。
- Torch 版输入在 CPU 时回退 native；要求张量 contiguous 且同 device/dtype。

### 4.4 mHC

- Pre-Op：多流 `x [B, T, n, C]` → 单流 `x_layer_in [B, T, C]` +
  `H_post [B, T, n]` + `H_res [B, T, n, n]`。
- Post-Op：单流 `layer_out` + 原多流 `x` + `H_post` + `H_res` → 新多流 `x_next`。
- 内部流程：`linear_and_reshape` → `Sinkhorn-Knopp`（双随机矩阵，减 max 防溢出）
  → `stream_aggregate`。
- **约束**：`C` 必须能被 128 整除（四个 Triton 公开入口已加显式
  `ValueError` 校验）；投影矩阵输出维度 `M = n*(n+2)` 必须是 32 的倍数
  （native 内有 assert）。
- **显存优化靠手写 VJP 强制重计算**：pre_op 反向在 kernel 内用未归一化的原始
  输入重跑 sinkhorn，前向只保存原始输入。**改 sinkhorn 数学必须同步改
  fwd+bwd 两个 kernel。**
- jax_triton 反向 grid 固定 `(total_bt, 1)`——消除原子操作的关键，勿改二维。
- triton 只替换 `mhc_pre_op_fused` 与 `mhc_post_op` 两个底层符号，高层封装
  始终共用 native 的 `linear_and_reshape`。

### 4.5 Pallas 后端（`jax_pallas_kernel.py` + `pallas_utils.py`）

- **只用公开稳定 Pallas API**：`pl.pallas_call` / `pl.BlockSpec` /
  `pl.program_id` / ref 索引 / `lax.fori_loop` / `jnp`。**禁止 import
  plgpu/pltriton 私有 API**，以兼容老版 jax（Triton lowering）、未来版本
  （Mosaic GPU lowering）与 TPU。
- kernel 结构：grid=(B, N)，每个 program 处理一个 (batch, head)；外层
  `fori_loop` 遍历 chunk、内层静态展开 `chunk_size` 步；数学与 `triton_kernel.py`
  逐行对应（MINI_BSZ=1 的角色由 grid 取代，kernel 内无需 batch mask 与
  64 位指针运算）。
- **后端选择是确定性能力探测，不做计时择优**：按偏好顺序（默认后端 →
  triton 后端）逐个编译探测，第一个能 lowering 的即为本机后端。jax < 0.9
  的 GPU pallas 只有 triton lowering，探测自然落到 triton；未来 triton 后端
  被移除后自然落到默认后端。可用 `RWKV_OPS_PALLAS_BACKEND=default|mgpu|triton`
  强制覆盖（调试用）。
- **autotune 只调已选定后端的性能参数**（按 shape key 缓存最优）：triton
  后端调 `num_warps × num_stages`；MGPU 调 `lowering_semantics` /
  `reduction_scratch_bytes`。`RWKV_OPS_PALLAS_AUTOTUNE=0` 关闭后用默认参数。
  （jax 0.10.x 的 MGPU lowering 对逐行动态索引有 128 元素向量约束，探测会
  失败并落回 triton lowering。）
- **warmup 约定（重要）**：`custom_partitioning` 即使在 eager 调用下也会 trace
  内层函数，因此每个 custom_vjp 入口（primal / `_fwd` / `_bwd`）必须先用真实
  数组调用对应 warmup（`ensure_config`）解析配置；被 trace 的路径只查缓存，
  取不到则用首个候选。**warmup 与 launcher 的参数顺序必须一致**（曾因
  `dy/sa/state_chkp` 顺序错位导致所有候选探测失败的隐蔽 bug）。
- SPMD 分片规则与对应 triton 封装完全一致（同一套 sharding rule 字符串与
  infer_sharding 辅助函数）。

---

### 4.6 Pallas 编写经验（从 Triton 翻译）

本节记录如何把已有的 Triton kernel 翻译成只使用公开 Pallas API 的 JAX kernel。
项目内参考文件：`rwkv_ops/rwkv7_kernel/jax_pallas_kernel.py`、
`rwkv_ops/rwkv7_sane_kernel/jax_pallas_kernel.py`、对应 `triton_kernel.py`、
`rwkv_ops/pallas_utils.py`。

#### 4.6.1 为什么要翻译而不是直接用 jax-triton

`jax-triton` 依赖额外的 `jax-triton` 包，且 Pallas Triton backend 在 JAX 0.11.0
已标记 deprecated（未来会移除）。Pallas 提供前端一致的 API，同一份 kernel 代码可由
JAX 自动 lowering 到 Triton（旧版）或 Mosaic GPU（新版），未来也更容易迁移到 TPU。
我们的原则是：**用 Pallas 公开 API 写 kernel，后端选择交给 `pallas_utils` 的能力探测**。

#### 4.6.2 核心语义对照表

| Triton | Pallas | 说明 |
|---|---|---|
| `@triton.jit` | 普通 Python 函数 + `pl.pallas_call` | kernel 函数接收 `Ref` 参数 |
| `tl.program_id(0)` | `pl.program_id(0)` | 网格索引 |
| `tl.arange(0, H)` | 不需要显式构造 | 通过 `ref[b, h, t, :]` 隐式得到 |
| `tl.load(ptr, mask=..., other=...)` | `ref[...]` 或 `pl.load` | 用 `Ref` 索引代替指针算术 |
| `tl.store(ptr, val, mask=...)` | `ref[...] = val` | 直接写入 |
| `for t in range(T)` | `jax.lax.fori_loop(0, T, body, init)` | 动态循环（状态显式传递） |
| `for j in range(CHUNK_LEN)` | 原生 `for j in range(CHUNK_LEN)` | chunk 内静态展开 |
| `tl.sum(x, axis=2)` | `jnp.sum(x, axis=1)` | 注意维度对应关系 |
| `tl.where(c, a, b)` | `jnp.where(c, a, b)` | 元素选择 |
| `tl.exp` / `tl.log` / ... | `jnp.exp` / `jnp.log` / ... | JAX 标准一元函数 |
| `tl.constexpr` | 闭包 / Python 常量 / 全局常量 | 通过闭包捕获或全局传入 |

#### 4.6.3 从 Triton 到 Pallas 的翻译步骤

1. **去掉指针和 batch mask**：把 `tl.load` / `tl.store` 换成 `Ref` 索引。如果 Triton 里用
   `b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)` 和
   `b_mask = b_range < B_BATCH` 处理多个 batch，在 Pallas 里把 batch 维放到 grid 中，
   每个 program 只处理一个 `(b, h)`。这样就不需要 64 位指针运算和 mask。
2. **调整维度**：Triton 中为了保持 warp 效率，state 通常是 `[MINI_BSZ, H, H]`；Pallas 中
   每个 program 只处理一个 sample/head，state 退化为 `[H, H]`。因此 Triton 里
   `axis=2` 的 reduce 在 Pallas 中变成 `axis=1`。
3. **循环结构**：外层 chunk 循环用 `jax.lax.fori_loop`（需要把 state 作为 carry 传递），
   内层 `chunk_size` 步用原生 `for` 静态展开。
4. **数值稳定**：Triton 里常用手写 `tanh` 的 exp 稳定形式；Pallas 直接用 `jnp.tanh` 即可，
   JAX 编译器会生成稳定实现。
5. **输出声明**：在 Pallas 中，所有输出通过 `pl.pallas_call` 的 `out_shape` 声明，
   kernel 函数接收对应 `Ref` 并写入。

#### 4.6.4 内存空间与 BlockSpec

Pallas 的 `BlockSpec` 替换了 Triton 的手动指针切片。本项目中所有输入/输出都是完整张量的
"整数组"视图，因此 `pallas_utils.whole_specs` 统一构造 `BlockSpec`：

- Triton backend / TPU：`memory_space=pl.ANY`
- Mosaic GPU backend：`memory_space=plgpu.MemorySpace.GMEM`

不要在内核里 import `plgpu`，所有后端相关逻辑集中在 `pallas_utils`。

#### 4.6.5 后端选择与 autotune

不要把 autotune 理解为"在 Triton 和 Mosaic GPU 之间选最快的"。后端由能力探测确定
（default → triton → mgpu），autotune 只调**已选定后端**的性能参数：

- Triton backend：`num_warps`、`num_stages`
- Mosaic GPU backend：`lowering_semantics`、`reduction_scratch_bytes`

#### 4.6.6 常见陷阱

- **axis 对应**：Triton `axis=2` → Pallas `axis=1`（因为少了 `MINI_BSZ` 维）。
- **mask 维度**：Triton 里 mask 常广播成 3D `[B, 1, 1]`；Pallas 里每个 program 只处理
  一个 `(b, h)`，mask 退化为标量或一维。
- **参数顺序**：warmup 的 `ensure_config` 与 launch 的输入参数顺序必须完全一致，
  否则 trace 路径会拿到错误配置（曾因此导致所有候选探测失败）。
- **不要在 traced 路径探测后端**：所有后端探测在 warmup 阶段完成并缓存，
  `launch` 在 trace 路径只查缓存。
- **Triton 手写 tanh**：直接替换为 `jnp.tanh`，不需要保留分段 exp 形式。
- **状态循环**：Pallas 的 `fori_loop` 必须有显式返回值，state 作为循环 carry。

#### 4.6.7 最小翻译 checklist

- [ ] 把 `tl.program_id` 换成 `pl.program_id`
- [ ] 去掉 `tl.arange` 和指针运算，改用 `Ref` 索引
- [ ] 把 `tl.load` / `tl.store` 换成 `ref[...]` 读写
- [ ] 把动态循环换成 `jax.lax.fori_loop`
- [ ] 把 chunk 内静态循环保留为原生 `for`
- [ ] 检查 reduction axis：Triton `axis=2` → Pallas `axis=1`
- [ ] 用 `jnp.tanh` 替换手写 tanh 稳定形式
- [ ] 用 `jnp.where` 替换 `tl.where`
- [ ] 声明所有输出 `out_shape`
- [ ] 在 `custom_vjp` 的 primal / `_fwd` / `_bwd` 中分别调用 warmup
- [ ] 确认 warmup 与 launch 的参数顺序一致

---

### 4.7 Gated DeltaNet（GDN）

GDN 基于带门控的 delta rule，与 RWKV-7 类似但输入侧使用独立的 `g`（decay
gate）和 `beta`（write gate），且对 `q`、`k` 做 L2 norm。

核心递推（见 `gdn_recurrent/native_keras_op.py`）：

```text
state_t = state_{t-1} * exp(g_t) + delta_t ⊗ k_t
kv_mem_t = sum_K(state_{t-1} * k_t)
delta_t = (v_t - kv_mem_t) * beta_t
y_t = sum_K(state_t * q_t)
```

- 状态 `state` 形状：`(B, H, K, V)`。`K` 为 key head size，`V` 为 value head size。
- 输入默认 layout：
  - `q, k`: `[B, T, H, K]`；`v`: `[B, T, H, V]`；
  - `g, beta`: `[B, T, H]`；
  - `initial_state`: `[B, H, K, V]` 或 `[1, H, K, V]`，float32。
- `q`、`k` 在算子内部做 L2 norm，并在最后按 `1/sqrt(K)` 缩放。
- `beta` **必须已在外部过 sigmoid**，算子内部不再重复做，以保证接口与训练框架
  的 gate 输出一致。
- `g` 是 log-space decay gate，越负遗忘越快。
- `gated_delta_net_chunk` 要求 `T % chunk_size == 0`，会在内部 pad 到 chunk_size
  整数倍，`chunk_size` 默认 16；CUDA 后端作为编译期常量需通过工厂函数指定，
  Triton/Pallas/native 后端可在调用时传入。
- `gated_delta_net_recurrent_sane` 的 `tau` 形状为 `[B, T//chunk_size, H]`、`mask`
  形状为 `[B, T//chunk_size]`，`chunk_size` 同样遵循上述规则（CUDA 工厂指定，
  其他后端调用时传入）。
- `gated_delta_net_recurrent`（无 SANE）接受 `chunk_size` 但忽略，仅保持签名一致；
  它与 `gated_delta_net_reference` 支持任意长度。
- chunkwise 实现先把 g 做 cumsum 得到 chunk 内 decay 矩阵，再用 Neumann 级数
  求 `(I - lower_triangular(k_beta k^T * decay))^{-1}`，最后按 recurrent 方式
  跨 chunk 传递 state。数值上必须与 `gated_delta_net_reference` 逐位一致。
- 当前 Phase 1 `gdn_chunk/` 仅提供纯 Keras native 实现；`gdn_recurrent/`
  在 PyTorch CUDA 后端已提供 Triton 前向 kernel 与 CUDA kernel，均包含训练、
  推理、单步 RNN 三个入口（`gated_delta_net_recurrent` / `..._inference` /
  `..._single_step`；CUDA 版位于 `torch_cuda_kernel/`，训练入口含反向），
  其余后端/框架仍回退 native。后续 cuda / triton / pallas 加速内核会按同样
  的目录约定扩展。

---

## 6. 构建与编译

### 6.1 包构建

- 构建后端 **hatchling**；运行时唯一依赖 `keras>=3.0`；测试可选依赖
  `pip install -e ".[test]"`（`pytest>=8.0`、`jax-triton>=0.3.1`）。
- **wheel 为纯 Python**（`py3-none-any`），CUDA/Triton 全部运行时 JIT 编译，
  发布物不含二进制 kernel。构建：`python -m build`。
- **版本号两处同步**：`pyproject.toml` 与 `rwkv_ops/__init__.py` 的 `__version__`。
- `MANIFEST.in` 包含所有 `.py`、CUDA/HIP/C++ 源、CMake 文件；排除 `build*/`、
  `dist/`、`.so` 等。新增编译产物类型要同步加排除规则；**不要手改 `dist/`**。

### 6.2 CUDA 扩展的懒编译

- CUDA 实现不在 whl 中预编译，**首次使用时编译**：
  - JAX：`cmake`（`jax_cuda_kernel*/CMakeLists.txt`）→ `build_<...>/wkv*.so`，
    经 `ctypes.CDLL` + `jax.ffi.register_ffi_target` 注册；
  - Torch：`torch.utils.cpp_extension.load`（用户级缓存 `~/.cache/torch_extensions`）。
- 编译产物写入源码树旁（`.gitignore` 已排除）；**修改 C++/CUDA 源码后需删除
  对应 `build*` 目录才会重编**。
- **nvcc_wrap**：所有 JAX FFI 经 `-DCMAKE_CUDA_COMPILER=rwkv_ops/cuda_tools/nvcc_wrap`
  注入，用 `cuda_tools/include/bits/mathcalls.h` 的 `#include_next` 绕过
  CUDA 13.1 与 glibc 2.41+ 的 `rsqrt/rsqrtf` 冲突。**不要修改系统 CUDA 头文件**；
  `nvcc_wrap` 需保持可执行权限且随包分发。

### 6.3 运行环境要求

- CMake 的 host C++ 编译器须与 CUDA 兼容；GCC 过新（如 GCC 15 + CUDA 13.1）时
  指定 `CC`/`CXX`/`CUDAHOSTCXX` 到 gcc-13 等（`tests/jax/conftest.py` 会自动探测）。
- ROCm：仅 RWKV-6 Torch 路径，`RWKV_USE_ROCM=1`。
- PyTorch CUDA 扩展依赖 `ninja`。

### 6.4 构建产物清理

- `clean_build_artifacts.clean_all()` 删除各 `jax_cuda_kernel*/build_*` 与
  `*.so`、ninja 日志、全部 `__pycache__`。
- `tests/conftest.py` 的 `pytest_sessionfinish` 自动调用——**每次跑完 pytest，
  JAX CUDA 测试下次会重新编译**（耗时是预期行为）。
- 新增带 FFI 的算子目录时，把它的 `build_*`/`*.so` 路径加进 `_CLEAN_PATTERNS`。

---

## 7. 测试规范

### 7.1 测试入口

```bash
pip install -e ".[test]"

# 各后端必须分进程运行（同一进程 Keras 后端只能设定一次）
pytest tests/torch -v
pytest tests/jax -v
pytest tests/numpy -v
pytest tests/tensorflow -v
pytest tests/openvino -v

# 跳过较重的 slow 测试
pytest tests/jax -v -m "not slow"
```

- 在**仓库根目录**运行（测试经根 conftest 的 `sys.path` 注入 import
  `tests.conftest`）。
- 子目录 conftest 用 `setdefault` 设 `KERAS_BACKEND` 与 `CUDA_VISIBLE_DEVICES`，
  无需手动设环境变量；`KERNEL_TYPE` 不进环境变量，全经 fixture 显式传参。
- markers：`torch` / `jax` / `numpy` / `tensorflow` / `openvino` / `slow`（编译类慢测试）。
- **文件名唯一性**：`tests/` 各子目录无 `__init__.py`，跨目录同名文件会
  `import file mismatch`；统一带后端前缀（`test_jax_*` / `test_torch_*` 等）。

### 7.2 共享 fixtures 与输入分布（`tests/conftest.py`）

- 根 conftest **刻意不 import torch/jax/keras/rwkv_ops**，避免收集阶段锁定后端。
- `rng` 固定种子 42；`rwkv7_shape = (5, 128, 6, 64)`（B, T, H, K）。
- `rwkv7_inputs`：`r/k/v ~ N(0,1)`；`a`、`b` 为同一 z 的 ±单位向量（b = -a）；
  `w = -softplus(w_raw) - 0.5`；`h0 ~ N(0,1)`。
- `rwkv7_sane_inputs`：加 `tau`，`x ~ N(7.0, 0.5)`、`tau = softplus(x) + 1.0`
  （tau ≈ 1000，近似恒等映射），形状 `[B, T//chunk_size, H]`；测试默认 chunk_size=16。
- `gdn_shape = (2, 128, 4, 64, 128)`（B, T, H, K, V）。
- `gdn_inputs`：`q/k/v ~ N(0,1)`；`g = -softplus(g_raw) - 0.5` 保证稳定衰减；
  `beta` 先过 sigmoid 使其落在 (0,1)；`h0 ~ N(0,1) * 0.1`。`q/k` 不做 L2 norm，
  由算子内部处理。
- **递推数值对拍必须用这些 fixture 的稳定分布**；随手造的随机数据会让
  delta-rule 递推指数发散，f32 参考自身误差都能到 1e5，无法用于判定。
- 断言工具 `assert_allclose_with_stats`：打印 exact/close/max/mean diff，
  判定只看 atol/rtol。

### 7.3 测试覆盖要求与容差惯例

- 新增/修改加速内核必须覆盖：前向输出与 final_state vs native；反向梯度
  vs native 自动微分；mask 全 1 / 全 0 / 随机 mask 等价性；
  `head_first=True/False` 两种 layout。
- **双精度覆盖（全项目默认强制）**：每个加速内核（Triton/CUDA/Pallas）的
  测试必须同时包含 **bf16和fp32用例**
  情况覆盖；两组共用同一组 numpy 输入，分别 cast 后按各自红线判定。
  若某内核只支持 bf16 I/O，fp32 组用例改为验证"fp32 输入警告并 cast"
  的行为，不得直接省略。
- SANE 类算子额外镜像覆盖：带/不带 mask 的前向+反向（含 tau 梯度）、无 mask
  警告 + None state、任意长度推理（T=34）、不规则 padding、head 轴 TP（jax）。
- 统一模式：同一组 numpy 输入分别喂 native 与加速算子，输入 cast bf16
  （state/tau/mask 保持 f32），loss 用 `mean(y²) + mean(state²)`。
- 精度红线（**全项目所有算子通用，任何测试不得突破**）：
  - **bf16：最大误差 1e-2**。
  - **fp32：最大 1e-3，目标 1e-4**。
  - 唯一例外：bf16 输入图下游的 fp32 叶子梯度（如 dbeta）带 bf16 输入
    噪声，最大可放宽到 2e-3。
- 各家族容差明细（均不得突破上述红线）：
  - RWKV-7 前向 y atol=1e-5 / rtol=1e-2（SANE 的 y 放宽到 atol=1e-4）；
    final_state atol=1e-5 / rtol=1e-3；反向 grad atol=7e-3 / rtol=1e-3
    （grad_b 放宽到 1e-2）。
  - GDN native：recurrent / chunkwise / reference 互相对齐，atol=1e-5 / rtol=1e-3；
    Triton/CUDA 前向与 native 对齐，atol=1e-4 / rtol=1e-3；bf16 放宽到 1e-2 / 1e-2。
  - DeltaNet native：互拍沿用 GDN 惯例——chunk 参与的对比输入 cast bf16
    （beta/state 保持 f32），atol=1e-2 / rtol=1e-3；recurrent vs reference /
    single_step / inference 对比保持 fp32，atol=1e-5 / rtol=1e-3。
  - DeltaNet Triton 反向：bf16 叶子（q/k/v）梯度 atol=1e-2 / rtol=1e-2；
    fp32 叶子梯度收紧——dbeta atol=2e-3 / rtol=1e-2（上述例外项），
    dh0 atol=1e-4 / rtol=1e-3。
  - RWKV-6 / mHC：一律 1e-2 / 1e-2。
- mHC 测试同时包含速度和显存基准。

---

## 8. 代码风格与协作规范

### 8.1 Python 代码

- `ruff check .` 与 `ruff format --check .` 必须全绿（ruff 无规则定制，
  下列约定靠 AGENTS.md 约束）。
- 禁止 `from ... import *`；禁止 lambda 赋值（`grid = lambda ...`），改用 `def`。
- 函数式入口：每个 kernel 的 Python 入口返回 `out` 或 `(out, final_state)`。
- 使用 `keras.ops` 编写后端无关逻辑；`torch.*` / `jax.*` / `tf.*` 只允许出现在
  对应后端绑定文件中。

### 8.2 注释与文档风格（**强制性约束**）

本节为**强制规范**，新增/修改代码必须遵守；评审时应以本节为准驳回不合规
注释。整体结构参照 Keras 3 的 docstring 风格，正文语言沿用项目现有的中文
（小节标题用英文：`Args` / `Returns` / `Raises` / `Examples`）。
禁止使用"------" "======="等一切形式的分割线  
这种# ===== 推理 Kernel（无 Mask） ===== 也不行  
应该是# 推理 Kernel（无 Mask）
同样的，cpp/cuda里/* -------------------- C 接口函数 -------------------- */
应该是// C 接口函数 或 /* C 接口函数 */
注释里禁止出现1. 2. 3. 4.等不便于修改的不符合人类注释风格的情况

**模块 docstring（一行制）**

- 只允许一行，声明模块职责，例如 `"""JAX 版 RWKV7 wkv kernel"""`。
- **禁止**写入迭代痕迹与实现沿革：如"延迟编译"、"与 Torch 版 1:1 对齐"、
  "重构后"、"新增"等。版本对齐关系属于 `AGENTS.md`，不进代码。

**函数 docstring（公开函数必须完整，结构按序）**

1. 一行描述；
2. 可选的详细段落；
3. 可选的 `Examples` 部分（**仅限公开 API 入口**，如
   `generalized_delta_rule`、`rwkv6_op`、`mhc_pre_op`）；
4. `Args` 部分：**每个参数必须带形状与 dtype**，关键约束写在该参数条目内
   （如 `tau: [B, T//chunk_size, H], float32, 必须 > 1`；`x: [B, T, n, C], C 必须
   被 128 整除`）；
5. `Returns` 部分：同样带形状与 dtype；
6. 可选的 `Raises` 部分：**所有显式 raise 的条件必须列出**（如
   `T % chunk_size != 0`、`M % 32 != 0`）。

私有函数（下划线开头）允许只写一行描述。

**函数 docstring 示例**

```python
def generalized_delta_rule_sane(r, w, k, v, a, b, tau, mask=None,
                              initial_state=None, output_final_state=True,
                              head_first=False):
    """带 State Anomaly Neutralization 的 RWKV-7 广义 delta 规则（chunkwise 训练版）。

    在 chunk 边界（每 chunk_size 个 token）按 mask 对 state 执行
    `state = tau * tanh(state / tau)`；输出始终基于 SANE 之前的 state。

    Args:
        r, w, k, v, a, b: [B, T, H, K], bfloat16。T 必须被 chunk_size 整除。
        tau: [B, T//chunk_size, H], float32。阈值，必须严格 > 1。
        mask: [B, T//chunk_size], float32 或 None。>0 的 chunk 边界执行 SANE；
            仅当 output_final_state=True 时生效。
        initial_state: [B, H, K, K] 或 [1, H, K, K], float32, 可选。
        output_final_state: bool, 是否返回最终 state。
        head_first: bool, 输入输出是否 head 维优先 ([B, H, T, K])。

    Returns:
        out: [B, T, H, K], 与输入同 dtype。
        final_state: [B, H, K, K], float32。
            output_final_state=False 时不返回；mask=None 时为 None。

    Raises:
        ValueError: T 不被 chunk_size 整除，或 tau/mask 形状不匹配。

    Examples:
        >>> y, state = generalized_delta_rule_sane(
        ...     r, w, k, v, a, b, tau, mask, initial_state=h0)
    """
```

**类 docstring（结构按序）**

1. 一行描述；2. 可选详细段落；3. 可选 `Examples`；4. `Args`（对应
   `__init__()` 参数）；5. 若是 Layer，另需 `Call arguments`（对应
   `call()` 参数）与 `Returns`，可选 `Raises`。

**inline comment：只写"为什么"，禁止写"改了什么"**

- **禁止**任何迭代痕迹注释：`【修改】`、`【修复】`、`【新增】`、
  `【稳定性修复 x/y】`、"暂时"、"之前是……" 等。历史由 git 管理。
- 允许并鼓励写**设计约束/不变量**：如 "MINI_BSZ=1：限制单 block 寄存器
  占用，规避 ROCm 编译器溢出问题"、"checkpoint 保存 SANE 之前的 state 供反向
  使用"。
- Section banner 统一为单行 `# ===== 标题 =====`，不带版本备注。

**禁止进入 docstring/注释的内容**

- 性能数字、基准结论（会过时；README 已有此教训）；
- 待办、吐槽、署名。

**CUDA / C++ 源码注释（`.cu` / `.cpp` / `.h` / `.cuh`）**

Keras 3 只规定 Python docstring；CUDA/C++ 按同等精神执行，同为强制约束：

- **文件头注释（一行制）**：`//` 或 `/* */` 一行声明 kernel 职责，例如
  `// RWKV-7 wkv forward/backward CUDA kernel`。禁止写入迭代沿革。
- **kernel 入口函数必须有块注释**，包含：
  - 一行描述（算的是什么）；
  - 每个指针参数的**形状、内存布局（row-major/strides）与 dtype**；
  - grid/block 的语义（如 "每个 block 处理一个 (batch, head)"）；
  - 编译期宏表（如 `_C_`=head_size、`_T_`=max_sequence_length、
    `CHUNK_LEN`=16），宏的取值约束写在内联注释里。
- **inline comment 与 Python 同规**：只写"为什么"（如"用 64 位整数做指针
  算术防止大 tensor 溢出"、"checkpoint 保存 SANE 之前的 state 供反向使用"、
  "blend 形式避免 warp 分支"），禁止 `【修复】`/`【修改】`/"暂时"类
  迭代痕迹。
- 数值相关的等价变换（如手写 tanh 的稳定形式）必须注明"为何不用朴素
  写法"。
- 禁止内容同 Python：性能数字、与 README 重复的教程、待办、吐槽、署名。
- 语言与 Python 侧一致：中文正文，结构词（如 `Args:` 可选）用英文。

**CUDA kernel 块注释示例**

```cpp
// RWKV-7 wkv 前向 CUDA kernel。
//
// 每个 block 处理一个 (batch, head)，顺序扫描 T 步，在每个 chunk 末尾
// 写出 SANE 之前的 state checkpoint。
//
// Args:
//   r, w, k, v, a, b: [B, H, T, K], bfloat16, row-major。
//   h0:  [B, H, K, K], float32, row-major。初始 state。
//   out: [B, H, T, K], bfloat16, row-major。输出 y。
//   sa:  [B, H, T, K], float32, row-major。反向所需中间量。
//   s_:  [B, H, T//chunk_size, K, K], float32, row-major。SANE 之前的 state checkpoint。
//
// 编译期宏:
//   _C_: head_size，必须被 4 整除。
//   CHUNK_LEN: chunk 长度，默认 16，可通过工厂函数或算子参数修改。
//
// 指针算术一律使用 64 位整数，防止大 tensor 时 32 位偏移溢出。
__global__ void wkv7_forward(...)
```

### 8.3 CUDA/Triton/Pallas 代码

- 编译期常量通过宏传入：`-D_N_=64 -D_T_=4096`。
- CUDA 指针算术使用 64 位整数；Triton 内核使用 `tl.int64`。
- 新增 Triton 内核时，JAX 和 PyTorch 的桥接文件需要同时更新，并共享同一套
  `triton_kernel.py`。
- Pallas 内核只用公开 API（见 §4.5）。

### 8.4 C/C++ 代码格式化

- C/CUDA 源文件（`.cu`、`.cuh`、`.cpp`、`.h`）使用 **clang-format** 以 LLVM style 格式化。
- 统一入口：`scripts/format_cpp.sh`（依赖 `clang-format`，可 `pip install clang-format`）。
- 提交/修改这些文件前，运行 `./scripts/format_cpp.sh`；IDE 可通过仓库根目录的 `.clang-format` 自动采用 LLVM style。
- 注意：`clang-format` 只控制排版，不改变注释内容与命名规范；注释仍须遵守 §8.2 的约束。

### 8.5 提交前检查清单

- [ ] `ruff check .` 与 `ruff format --check .` 全绿。
- [ ] `.cu/.cuh/.cpp/.h` 文件已运行 `./scripts/format_cpp.sh`（LLVM style）。
- [ ] 版本号两处同步（`pyproject.toml` + `rwkv_ops/__init__.py`）。
- [ ] 新增编译产物已加入 `.gitignore` 排除 / `MANIFEST.in` 包含规则。
- [ ] 新增/修改的算子已在 `rwkv_ops/__init__.py` 正确暴露。
- [ ] 对应后端测试已补充（文件名带后端前缀），并验证与 native 数值一致。
- [ ] Triton kernel 改动已同步检查 jax/torch 两个桥接。
- [ ] 新 FFI 算子的构建目录已加入 `clean_build_artifacts._CLEAN_PATTERNS`。
- [ ] 已更新 `AGENTS.md`、`README.md`、`ENREADME.md` 的支持矩阵。

---

## 9. 常见陷阱

1. **JAX RWKV6 `cuda` 后端用 `jax.ffi`**，需要 JAX >= 0.4.31；旧版 XLA
   custom-call 代码已移除。
2. **RWKV7 Triton 后端目前主要验证 HEAD_SIZE=64**，其他 head_size 请用 `cuda`。
3. **chunkwise RWKV7 要求序列长度能被 chunk_size 整除**，否则可能静默出错或触发
   未定义行为。
4. **RWKV6 的 `max_sequence_length` 是编译期常量**，修改后必须删除旧 build
   目录重新编译。
5. **mHC 的 `C` 必须被 128 整除**（Triton 入口已加显式校验）。
6. **不要直接修改 `dist/` 或 `build/` 目录中的内容**。
7. **`custom_partitioning` 在 eager 下也会 trace 内层函数**：任何运行期探测
   （如 pallas autotune）必须在 trace 之外完成（见 §4.5 warmup 约定）。
8. **JAX CUDA 测试每次会话后 `.so` 被自动清理**，下次重编译是预期行为。
9. **rwkv6 的 `KERNEL_TYPE="triton"` 静默回退 native**，不报错。
10. **递推对拍必须用 fixture 的稳定输入分布**（见 §7.2），随手造的随机数据
    会让递推指数发散，得出"实现错了"的假结论。
11. **JAX-Triton 桥接的 `*args` 顺序必须与 Triton kernel 签名完全一致**：
    Triton kernel 里 `scale` 等标量参数的位置（在 `B,H,T,K,V` 之前还是之后）
    会直接决定编译器如何解释 grid 与指针算术。`gdn_chunk` 曾因把 `scale`
    放在维度标量之前而导致 CUDA illegal address / 梯度 NaN。新增 kernel
    时务必逐一对照 `triton_kernel.py` 里的 torch 封装顺序。
12. **L2 norm 反向 kernel 需要传入原始输入而非归一化结果**：
    `gdn_chunk` 的 Triton L2 norm 反向公式基于原始输入推导，传归一化后的
    `q2/k2` 会在 fp32 下产生系统性偏差。JAX/Torch 封装都应在 forward 中额外
    保存原始 q/k 供 backward 使用。
13. **显式传 `BK`/`BV` 等 `tl.constexpr` 时，值必须在 autotune config 中存在**：
    否则 autotune prune 会报 "No valid autotuner configs"。`fwd_h` /
    `bwd_dhu` 的 autotune config 只含 `BV=64`，不要传 `BV=128`。

---

## 10. 扩展指南

若新增一种算子：

1. 在 `rwkv_ops/` 下新建目录，至少包含 `__init__.py` 和 `native_*.py`。
2. 在 `__init__.py` 中实现工厂函数，按 `KERNEL_TYPE` + `keras.config.backend()`
   分发（缺硬件静默回退 native）。
3. 为每个加速后端编写桥接文件，共享同一套 Triton/CUDA 内核；Pallas 后端复用
   `pallas_utils.py`。
4. 在 `rwkv_ops/__init__.py` 中导入并实例化、暴露。
5. 在 `tests/<backend>/` 中添加测试（文件名带后端前缀），fixture 挂进对应
   conftest（jax 侧按 `xxx_jax_op` / `xxx_jax_triton_op` / `xxx_jax_pallas_op`
   命名）。
6. 新 FFI 算子的构建目录加入 `clean_build_artifacts._CLEAN_PATTERNS`。
7. 更新本 `AGENTS.md`、`README.md`、`ENREADME.md` 中的支持矩阵。

---

## 11. JAX-Triton 桥接教程

> 目标：用 JAX 调用已有的 Triton kernel，并保证与 PyTorch / native 输出一致。
> 项目内参考实现：`rwkv_ops/rwkv7_kernel/jax_triton_kernel.py`、
> `rwkv_ops/rwkv7_sane_kernel/jax_triton_kernel.py`、
> `rwkv_ops/mhc_kernel/jax_triton_op/`。
> 依赖 `jax-triton>=0.3.1`，通过 `pip install -e ".[test]"` 安装。

### 11.1 前置说明

- 本文档**不教如何写 Triton kernel**，只教如何在 JAX 中做**桥接封装**。
- 核心难点：JAX 侧需要显式声明输出形状、常量参数、grid，并处理 PyTorch 侧隐式完成的事情（如 `None` 输入、内存初始化、SPMD 分片等）。
- 所有桥接文件共享同一个 `triton_kernel.py`，因此修改 kernel 数学时必须同步检查 JAX 与 PyTorch 两侧桥接。
- **`jt.triton_call` 的 `*args` 顺序必须与 Triton kernel 签名逐参数对齐**，包括 `scale` 这类标量张量在维度标量 `B,H,T,K,V` 之前还是之后。顺序错位不会报错，但会把标量解释成维度、把维度解释成标量，导致 CUDA illegal address、NaN 或静默错误。新增反向 kernel 时建议直接复制 torch 侧 `_kernel[grid](...)` 里的参数顺序。

### 11.2 基础调用范式

Triton kernel 的签名通常可分为三类参数：

- 输入张量：按顺序传入 `jt.triton_call` 的 `*args`。
- 输出张量：在 `out_shape` 中声明，顺序与 kernel 的输出指针对应。
- `tl.constexpr` 编译期常量：以 `**kwargs` 形式传入。

以 `rwkv7_sane_kernel/jax_triton_kernel.py` 的前向调用为例：

```python
import jax
import jax.numpy as jnp
import jax_triton as jt

from .triton_kernel import rwkv7_sane_fwd_kernel

def _wkv7_sane_fwd_triton_call(r, w, k, v, a, b, tau, h0, chunk_size: int):
    B, N, T, H = r.shape
    dtype = r.dtype
    chunk_num = T // chunk_size

    # out_shape 列表顺序与 kernel 的输出指针 OUT / SA_OUT / STATE_CHKP 一一对应
    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, H), dtype),
        jax.ShapeDtypeStruct((B, N, T, H), jnp.float32),
        jax.ShapeDtypeStruct((B, N, chunk_num, H, H), jnp.float32),
    ]

    # grid 与 Torch 侧保持一致
    def grid(meta):
        return ((B + meta["MINI_BSZ"] - 1) // meta["MINI_BSZ"], N)

    out, sa_out, state_chkp = jt.triton_call(
        r, w, k, v, a, b, tau, h0,  # 输入张量，顺序严格对齐 kernel 签名
        B, N, T,                     # 标量（kernel 里按指针/值读取均可）
        kernel=rwkv7_sane_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        H_SIZE=H,                    # tl.constexpr
        CHUNK_LEN=chunk_size,        # tl.constexpr
    )
    return out, sa_out, state_chkp
```

要点：

- `*args` 顺序必须严格对齐 kernel 的输入指针签名。
- `out_shape` 是 `jax.ShapeDtypeStruct(shape, dtype)` 的列表，长度与输出指针数量一致。
- `grid` 与 Torch 侧完全一致；如果 kernel 使用 `@triton.autotune`，`meta` 会包含 `MINI_BSZ` 等自动调参产生的值。
- 所有 `tl.constexpr` 通过 `**kwargs` 传入，不要在 `*args` 中传。

### 11.3 装饰器处理

如果 kernel 同时被 `@triton.heuristics` 和 `@triton.autotune` 包装，直接传 `kernel=xxx` 可能报错：

```text
'kernel' must be a Triton `JITFunction`, `Heuristics` or `Autotuner`.
```

原因：`jax-triton` 的校验无法识别被多层装饰器包裹后的对象。解决方法是取最底层 `JITFunction`：

```python
kernel=some_kernel.fn   # 跳过 heuristics，指向 autotune 或底层函数
```

同时需要手动补齐 heuristics 产生的参数：

```python
jt.triton_call(
    ...,
    kernel=some_kernel.fn,
    USE_FINAL_STATE_GRADIENT=True,
    USE_INITIAL_STATE=False,
)
```

> 项目当前 RWKV-7 / RWKV-7-SANE / mHC 的共享 kernel 只使用了 `@triton.autotune`，因此可以直接传 `kernel=rwkv7_fwd_kernel` 等，无需 `.fn`。

### 11.4 可选输入与可选输出

`jax-triton` 的 `*args` 和 `out_shape` 都不能出现 `None`。对于 PyTorch 侧可为 `None` 的参数（如 `initial_state`、`dht`），JAX 侧需显式构造零张量。

参考 `rwkv7_kernel/jax_triton_kernel.py` 的反向封装：

```python
def _bwd(res, grads):
    r, w, k, v, a, b, sa_out, state_chkp = res
    dy, dht = grads
    dy = jnp.asarray(dy, jnp.bfloat16)
    if dht is None:
        B, N, T, H = r.shape
        dht = jnp.zeros((B, N, H, H), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    dr, dw, dk, dv, da, db, dh0 = _wkv7_bwd_spmd(
        r, w, k, v, a, b, dy, sa_out, state_chkp, dht
    )
    return dr, dw, dk, dv, da, db, dh0
```

`out_shape` 中也不能放 `None`。即使某个输出在特定分支不需要，也要给它一个合法的 `ShapeDtypeStruct`；通常通过常量参数控制 kernel 内部是否真正写入。

### 11.5 原子操作与内存初始化

当 kernel 内部使用 `tl.atomic_add` 等原子操作时，输出内存必须预先初始化为 0，否则可能复用 XLA 分配的脏内存导致结果错误。`jax-triton` 提供 `zeroed_outputs` 参数指定哪些输出需要清零：

```python
jt.triton_call(
    ...,
    out_shape=[jax.ShapeDtypeStruct(shape, dtype)],
    zeroed_outputs=(0,),   # 第 0 个输出在 kernel 执行前清零
)
```

> 项目当前通过设计避免原子操作：例如 mHC 反向 kernel 把 grid 固定为 `(total_bt, 1)`，让不同 program 写不同输出位置，从而不需要原子加；RWKV-7 系列 kernel 的输出由每个 block 独占写回，也不依赖 `zeroed_outputs`。新增使用原子操作的 kernel 时才需要显式传入该参数。

### 11.6 custom_vjp 与 custom_partitioning

JAX 桥接需要同时支持自动微分和 SPMD 分片，因此通常把 `jt.triton_call` 包在三层结构里：

- 最内层：`jt.triton_call(...)` 的 launcher。
- 中间层：`@custom_partitioning` + `def_partition(...)`，声明 sharding 规则。
- 最外层：`@jax.custom_vjp` + `defvjp(...)`，声明反向传播。

参考 `rwkv7_sane_kernel/jax_triton_kernel.py`：

```python
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

# Einsum 风格的 sharding rule：同一字母表示同一轴，支持 head 维 TP
FWD_RULE = (
    "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n c, b n h h -> "
    "b n t h, b n t h, b n c h h"
)

def _fwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    return (
        _sharding_like_q(qs),
        _sharding_like_q(qs),
        _sharding_for_state(qs),
    )

def _create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)
        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings
    return partition

# 1. launcher
def _wkv7_sane_fwd_triton_call(r, w, k, v, a, b, tau, h0):
    ...
    return jt.triton_call(...)

# 2. custom_partitioning
@custom_partitioning
def _wkv7_sane_fwd_spmd(r, w, k, v, a, b, tau, h0):
    return _wkv7_sane_fwd_triton_call(r, w, k, v, a, b, tau, h0)

_wkv7_sane_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=_create_partition(_wkv7_sane_fwd_triton_call),
)

# 3. custom_vjp
@jax.custom_vjp
def rwkv7_sane_kernel_triton(r, w, k, v, a, b, tau, h0):
    out, sa_out, state_chkp = _wkv7_sane_fwd_spmd(...)
    return out, final_state

rwkv7_sane_kernel_triton.defvjp(_fwd, _bwd)
```

注意 `custom_partitioning` 定义时需要 `static_argnums` 来处理非张量参数。参考 `mhc_kernel/jax_triton_op/mhc_pre_op.py`：

```python
mhc_pre_op_fwd_spmd = custom_partitioning(
    _mhc_pre_op_fwd_spmd_impl, static_argnums=(3, 4)
)
```

`num_iters` 和 `eps` 作为静态参数，不会进入分片规则的追踪。

### 11.7 布局、步幅与数值一致性

- **连续布局**：`jax-triton` 默认按连续内存做指针偏移。传入 strided 张量可能得到错误结果。RWKV-7 系列在转置到 head-first 后必须保证 contiguous；Torch 侧会用 `.contiguous()` 处理，JAX 侧通常在 `_transpose_head` 后已通过 `jnp.transpose` 得到连续布局，但仍需注意不要传非连续视图。
- **步幅参数**：mHC kernel 需要显式传入各张量的 stride，例如 `stride_x_bt`、`stride_x_n`、`stride_x_c` 等。JAX 侧用 `jt.strides_from_shape(shape)` 计算后传入，保证与 kernel 内部的指针算术一致。
- **dtype 对齐**：Triton kernel 通常 `bfloat16` I/O、`float32` 内部累加。JAX 侧进 kernel 前需要把 state / tau / mask 等 cast 到 `jnp.float32`，输出后再 cast 回原始 dtype。
- **数值对拍**：新增 kernel 后，先用 `native_keras_op.py` 的输出作为 ground truth，对同一组 numpy 输入分别跑 native 与 triton 版本，比较前向输出、final_state 与反向梯度。

### 11.8 通用调用模板

```python
import jax
import jax_triton as jt

def call_triton_kernel(kernel, inputs, out_shapes, constants, grid, zeroed=None):
    """通用 JAX-Triton 调用模板。

    Args:
        kernel: Triton JITFunction（如有多层装饰器请传 .fn）。
        inputs: list[JaxArray]，对应 kernel 的输入指针。
        out_shapes: list[ShapeDtypeStruct]，顺序对应 kernel 的输出指针。
        constants: dict，对应 kernel 的 tl.constexpr 参数。
        grid: tuple 或 callable，launch grid。
        zeroed: tuple[int]，需要在 kernel 执行前清零的输出索引。

    Returns:
        单个 jax.Array 或 tuple，与 out_shapes 一一对应。
    """
    return jt.triton_call(
        *inputs,
        kernel=kernel,
        out_shape=out_shapes,
        grid=grid,
        zeroed_outputs=zeroed or (),
        **constants,
    )
```

核心记住三点：**输入顺序严格对齐**、**输出形状显式声明**、**特殊参数（None / 装饰器 / 原子操作 / 静态参数）显式处理**。

---

## 12. JAX 自定义算子分片教程

> 目标：让 Triton / Pallas / CUDA-FFI 自定义算子在 JAX SPMD 训练下正确传播分片，
> 支持 DP（batch 维并行）与 TP（head 维并行）。
> 参考：[JAX FFI 外部函数接口与分片](https://jax.net.cn/en/latest/ffi.html)、
> `rwkv_ops/rwkv7_kernel/jax_pallas_kernel.py`、
> `rwkv_ops/rwkv7_kernel/jax_cuda_kernel/wkv7_jax.py`、
> `rwkv_ops/rwkv7_sane_kernel/jax_pallas_kernel.py`、
> `rwkv_ops/rwkv7_sane_kernel/jax_cuda_kernel/wkv7_sane_jax.py`。

### 12.1 为什么自定义算子需要显式分片

普通 JAX 算子（`jnp.matmul`、`jax.lax.scan` 等）在 `jit(..., in_shardings=..., out_shardings=...)` 下会自动被 XLA 重新分区（reshard / all-gather / all-reduce）。但自定义算子对 XLA 是黑盒：

- 不做声明的 FFI 调用会把分片输入 `all-gather` 成全量，在每个设备上跑完整 kernel，再切片回分片输出，通信开销巨大。
- `shard_map` 可以把手动分区边界内的 FFI 调用限制在本地 shard 上执行，无需通信。
- `custom_partitioning` 可以把分区逻辑注册到 XLA 编译器，让 `jit` 在自动并行化时直接生成分区后的自定义调用。

本项目所有 JAX 加速算子（Pallas / Triton / CUDA FFI）统一使用 `custom_partitioning` 方案，
使其在常规 `jax.jit(..., in_shardings=..., out_shardings=...)` 流程中自动支持 DP/TP。

### 12.2 基础概念速览

```python
import jax
from jax.sharding import NamedSharding, PartitionSpec, Mesh

# 构造一个 4 卡 mesh，轴名为 "data"。
mesh = jax.make_mesh((4,), ("data",))

# 对 4 维张量 [B, T, H, K] 在 batch 维切分。
sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None))

x = jax.device_put(x, sharding)
```

- `PartitionSpec` 中的 `None` 表示该维度在所有设备上 replicate。
- 同一 mesh 轴名出现在多个张量的不同维度上时，XLA 会在必要时做 all-gather / all-reduce。
- 自定义算子需要告诉 XLA "这个张量的 batch 维可以切、time 维不能切、state 维需要 replicate"。

### 12.3 核心方案：custom_partitioning + Einsum sharding_rule

项目内所有 JAX 自定义算子采用同一套样板：

```python
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec
import jax.tree_util as jtu

# Einsum 风格规则：同一字母的维度必须同形且可一起切分。
FWD_RULE = (
    "b n t h, b n t h, b n t h, b n t h, b n t h, b n t h, b n h h -> "
    "b n t h, b n t h, b n c h h"
)

def _fwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    return (
        _sharding_like_q(qs),       # 输出 y 与输入 q 同形
        _sharding_like_q(qs),       # 输出 sa 与输入 q 同形
        _sharding_for_state(qs),    # 输出 state checkpoint 形状不同，需要重建
    )

@custom_partitioning
def _wkv7_fwd_spmd(r, w, k, v, a, b, h0):
    return _wkv7_fwd_pallas_call(r, w, k, v, a, b, h0)

_wkv7_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=create_partition(_wkv7_fwd_pallas_call),
)
```

`create_partition` 来自 `rwkv_ops/pallas_utils.py`，作用是在分区后的本地 mesh 上直接执行 `impl_fn`，
并把输入/输出的 `NamedSharding` 透传给 XLA：

```python
def create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)
        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings
    return partition
```

### 12.4 维度字母约定与可切分维度

本项目规则统一使用以下字母：

| 字母 | 含义 | 是否可切 |
|---|---|---|
| `b` | batch | ✅ 可 DP |
| `n` / `h` | head | ✅ 可 TP |
| `t` | time | ❌ 不能切（扫描需要完整 T） |
| `k` / `m` | head_size | ❌ 不能切（state 需要完整 head_size） |
| `c` | chunk | ❌ 不能切（chunk 是扫描的聚合单位） |

> 只允许切 batch 或 head 维；切 time 或 head_size 维会静默算错。

以 `rwkv7_kernel/jax_pallas_kernel.py` 为例：

```python
FWD_RULE = "b n t h, ..., b n h m -> b n t h, b n t h, b n c h m"
```

- 所有输入的 `b` 与 `n` 用同一字母，表示 batch 维与 head 维会随输入一起切分。
- `t`、`h`（time 与 head_size）用独立字母且不与输出混用，表示 replicate。
- state checkpoint `(B, N, C, H, H)` 的 sharding 用 `(spec[0], spec[1], None, spec[3], spec[3])`，
  即 batch/head 可切，chunk 与两个 head_size 维 replicate。

### 12.5 输出 sharding 不是输入同形时如何重建

`state`、`final_state`、`dtau` 等输出的形状与输入不同，不能简单继承输入 sharding。
项目内使用从输入 `q` 的 `NamedSharding.spec` 重建的方法：

```python
def _q_spec(qs):
    spec = getattr(qs, "spec", None)
    if spec is None or len(spec) != 4:
        return None
    return spec

def _sharding_for_state(qs):
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(
        qs.mesh,
        PartitionSpec(spec[0], spec[1], None, spec[3], spec[3])
    )

def _sharding_for_final_state(qs):
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(
        qs.mesh,
        PartitionSpec(spec[0], spec[1], spec[3], spec[3])
    )
```

在 CUDA FFI 版（`wkv7_jax.py`）中，输入 layout 是 `[B, T, H, K]`，
因此 `spec` 的下标映射与 Pallas 版略有不同：

```python
def _sharding_for_state(qs):
    spec = _q_spec(qs)  # [B, T, H, K]
    return NamedSharding(
        qs.mesh,
        PartitionSpec(spec[0], spec[2], None, spec[3], spec[3])
    )
```

> 关键原则：拿到输入 `NamedSharding` 后，按输出维度顺序从 `spec` 中取出对应轴名，
> 无对应维度或不能切的维度置 `None`。

### 12.6 Pallas 与分片的关系

Pallas kernel 的 `grid=(B, N)` 天然对应 "每个 program 处理一个 (batch, head)"：

```python
def _rwkv7_fwd_kernel(r_ref, w_ref, ..., h0_ref, o_ref, sa_ref, chkp_ref):
    b = pl.program_id(0)
    h = pl.program_id(1)
    state = h0_ref[b, h].astype(jnp.float32)
    ...
```

- 当 batch 或 head 被切分时，每个设备只会拿到本地 `(B_local, N_local)` 子集，
  grid 自动缩小，无需在 kernel 内写额外逻辑。
- `pallas_call` 的输入/输出 `BlockSpec` 在 `pallas_utils.whole_specs` 中统一构造，
  使用 `pl.BlockSpec(memory_space=pl.ANY)`（Triton/TPU）或 `plgpu.MemorySpace.GMEM`（MGPU）。
- `custom_partitioning` 包裹后，XLA 会在 mesh 上为每个设备 launch 对应子 grid。

Pallas 版需要注意 warmup：

```python
def _wkv7_fwd_warmup(r, w, k, v, a, b, h0):
    B, N, T, H = r.shape
    ensure_config(
        "wkv7_fwd", _rwkv7_fwd_kernel,
        _fwd_out_shape(r), (B, N),
        (r, w, k, v, a, b, h0),
    )
```

`custom_partitioning` 在 eager 下也会 trace 内层函数，因此必须先用真实数组调用 `ensure_config`
解析后端配置；被 trace 时只查缓存，否则会用首个候选导致探测失败。

### 12.7 CUDA FFI 版的分片封装

CUDA FFI 通过 `jax.ffi.ffi_call` 调用外部 `.so`。分片封装方式与 Pallas/Triton 完全一致：

```python
def _wkv7_kernel_impl(w, q, k, v, a, b, h0):
    B, T, H, K = q.shape
    dtype = q.dtype
    chunk_num = int(T // CHUNK_LEN)
    out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
    s_type = jax.ShapeDtypeStruct((B, H, chunk_num, K, K), jnp.float32)
    sa_type = jax.ShapeDtypeStruct((B, T, H, K), jnp.float32)

    return jax.ffi.ffi_call(
        "wkv7_fwd", (out_type, s_type, sa_type),
        vmap_method="broadcast_all",
    )(w, q, k, v, a, b, h0)

@custom_partitioning
def _wkv7_kernel(w, q, k, v, a, b, h0):
    return _wkv7_kernel_impl(w, q, k, v, a, b, h0)

_wkv7_kernel.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=_create_partition(_wkv7_kernel_impl),
)
```

注意：

- `vmap_method="broadcast_all"` 让 FFI 调用在 batch 维被 `vmap` 时直接广播处理，
  与 `custom_partitioning` 的 batch 切分语义一致。
- 后端 C++ kernel 必须能处理任意 `(B_local, H_local)` 子集；
  本项目 CUDA kernel 的 grid 也是 `(batch_blocks, head_blocks)`，与 Pallas 相同。
- `jax.ffi.register_ffi_target(..., platform="CUDA")` 注册的符号名
  必须与 `ffi_call("symbol_name", ...)` 完全一致。

### 12.8 Triton 版的分片封装

Triton 版（`rwkv7_kernel/jax_triton_kernel.py`）同样用 `custom_partitioning`，
只是底层调用 `jax_triton.triton_call` 而不是 `pl.pallas_call` / `ffi_call`：

```python
@custom_partitioning
def _wkv7_fwd_spmd(r, w, k, v, a, b, h0):
    return _wkv7_fwd_triton_call(r, w, k, v, a, b, h0)

_wkv7_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=_create_partition(_wkv7_fwd_triton_call),
)
```

Triton kernel 的 grid 为 `((B + MINI_BSZ - 1) // MINI_BSZ, N)`，
即第一个维度按 batch block 切分，第二个维度按 head 切分；
`custom_partitioning` 会保证每个设备只拿到它负责的那部分 grid。

### 12.9 mask / tau 等特殊张量的分片

在 RWKV-7-SANE 中：

- `tau` 形状 `[B, T//chunk_size, H]`，带 head 维，可随 head 轴 TP 切分。
- `mask` 形状 `[B, T//chunk_size]`，所有 head 共享，无 head 维；规则中用 `b c`，
  因此 head 轴 TP 时会自动 replicate。

以 SANE Pallas 规则为例：

```python
FWD_MASK_RULE = (
    "b n t h, b n t h, ..., b n c, b c, b n h h -> b n t h, b n t h, b n c h h"
)
```

`b n c` 的 `tau` 与 `b c` 的 `mask` 区别就在于 head 维：

```python
def _sharding_for_tau(qs):
    spec = _q_spec(qs)
    return NamedSharding(
        qs.mesh,
        PartitionSpec(spec[0], spec[1], None)  # [B, N, C]
    )
```

### 12.10 单步算子（T=1）的分片

单步 RWKV-7 / RWKV-7-SANE CUDA kernel（用于 decode）同样支持 DP 与 TP：

```python
FWD_RULE = "b h k v, b h k, b h k, b h k, b h k, b h k, b h k v -> b h k v, b h k v"
```

- 输入输出都是 `(B, H, K, V)` 级别的 state，没有 time 维。
- batch 与 head 可切，head_size 与 value_size 必须 replicate。
- SANE 单步的 `tau` 为 `[B, H]`、`do_sane` 为 `[B]`，规则同样覆盖。

### 12.11 结构验证：单卡 1-device mesh

多卡环境不是人人都有，但单卡可以用 1-device mesh 做结构验证：

```python
import jax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

mesh = jax.make_mesh((1,), ("data",))
sharding = NamedSharding(mesh, P("data", None, None, None))

x = jax.device_put(x, sharding)
y, state = jax.jit(op, out_shardings=(sharding, None))(x)
```

验证点：

- `jit` 编译通过，不报错。
- 输入/输出的 `sharding.spec` 符合预期。
- 数值与不分片版本完全一致。

真正的 all-gather / all-reduce 行为需要在多卡环境复核，但结构验证可以提前发现规则拼写、
输出维度映射、mask 缺失等大部分问题。

### 12.12 常见陷阱

- **时间维被切**：规则中若把 `t` 写成可切字母，编译不报错但结果错误，因为 scan 需要完整 T。
- **state 维映射错**：`final_state [B, H, K, K]` 的 PartitionSpec 写成 `(spec[0], spec[1], spec[2], spec[2])`
  而 `spec[2]` 是 time 维时，会把 time 轴错误地当作 head_size 轴切分。
- **CUDA 与 Pallas 的 spec 下标不同**：CUDA FFI 输入是 `[B, T, H, K]`，Pallas/Triton 内部转成 `[B, N, T, H]`，
  `_sharding_for_state` 的下标必须分别对应。
- **忘记 warmup**：Pallas 版在 `custom_vjp` 的 primal/fwd/bwd 三个入口都要先调用对应 `ensure_config`，
  否则 trace 路径会拿到错误的后端配置。
- **规则字母重复导致报错**：Einsum 规则中同一字母不能出现在输出中多次，除非对应维度确实同形；
  必要时用 `m` / `k` 等额外字母区分两个 head_size 维。
- **FFI 未设置 `vmap_method`**：默认行为可能在 batch 切分下回退为 `scan`，性能或正确性受影响。

### 12.13 快速检查清单

新增 JAX 自定义算子时，按以下清单检查分片：

- [ ] 确定可切维度（通常只有 batch/head）。
- [ ] 为前向/反向/推理分别写出 Einsum sharding_rule。
- [ ] 实现 `infer_sharding_from_operands`，对输出形状不同的张量重建 NamedSharding。
- [ ] 用 `create_partition`（或等价的 `_create_partition`）包装 impl_fn。
- [ ] 在 `custom_partitioning` 上调用 `def_partition` 注册三者。
- [ ] Pallas 版补充 `ensure_config` / `launch` 调用，保证 eager 与 traced 路径配置一致。
- [ ] FFI 版设置 `vmap_method="broadcast_all"`。
- [ ] 单卡 1-device mesh 结构验证通过。
- [ ] 多卡环境下对比 native 数值一致。

---

## 13. 相关文件速查

| 文件 | 作用 |
|---|---|
| `rwkv_ops/__init__.py` | 包入口、环境变量解析、API 暴露（import 即实例化全部算子） |
| `rwkv_ops/pallas_utils.py` | Pallas 后端公共机制：候选配置、autotune、SPMD 辅助 |
| `rwkv_ops/rwkv7_kernel/__init__.py` | RWKV-7 后端分发器 |
| `rwkv_ops/rwkv7_kernel/native_keras_op.py` | RWKV-7 原生参考实现 |
| `rwkv_ops/rwkv7_kernel/triton_kernel.py` | RWKV-7 共享 Triton 内核 |
| `rwkv_ops/rwkv7_kernel/jax_triton_kernel.py` | RWKV-7 JAX-Triton 桥接 |
| `rwkv_ops/rwkv7_kernel/jax_pallas_kernel.py` | RWKV-7 Pallas 内核 |
| `rwkv_ops/rwkv7_sane_kernel/` | SANE 版，结构与 rwkv7_kernel 完全平行 |
| `rwkv_ops/mhc_kernel/jax_triton_op/` | mHC JAX-Triton 桥接 |
| `rwkv_ops/gdn_chunk/native_keras_op.py` | GDN chunkwise 原生参考实现 |
| `rwkv_ops/gdn_chunk_sane/native_keras_op.py` | GDN chunkwise SANE 原生参考实现 |
| `rwkv_ops/gdn_chunk_sane/triton/chunk_h.py` | GDN chunkwise SANE 状态前向 Triton kernel |
| `rwkv_ops/gdn_chunk_sane/triton/chunk_bwd_dhu.py` | GDN chunkwise SANE 状态反向 Triton kernel |
| `rwkv_ops/gdn_recurrent/native_keras_op.py` | GDN recurrent 原生参考实现 |
| `rwkv_ops/gdn_recurrent/triton_kernel.py` | GDN recurrent 共享 Triton 内核 |
| `rwkv_ops/gdn_recurrent/torch_triton_kernel.py` | GDN recurrent PyTorch Triton 桥接 |
| `rwkv_ops/gdn_recurrent/torch_cuda_kernel/` | GDN recurrent PyTorch C++/CUDA 扩展（训练含反向 / 推理 / 单步） |
| `rwkv_ops/gdn_recurrent/jax_cuda_kernel/` | GDN recurrent JAX FFI CUDA（训练含反向 / 推理 / 单步，按 (K,V,chunk) 懒编译） |
| `rwkv_ops/gdn_recurrent_sane/torch_cuda_kernel/` | GDN recurrent SANE PyTorch C++/CUDA 扩展（训练含反向含 dtau / 推理 / 单步） |
| `rwkv_ops/gdn_recurrent_sane/jax_cuda_kernel/` | GDN recurrent SANE JAX FFI CUDA（训练含反向 / 推理 / 单步，按 (K,V,chunk) 懒编译） |
| `rwkv_ops/delta_net_chunk/native_keras_op.py` | DeltaNet chunkwise 原生参考实现 |
| `rwkv_ops/delta_net_chunk/triton/` | DeltaNet chunkwise 共享 Triton 内核（l2norm / intra / wy / chunk_h / chunk_o / 反向各 kernel） |
| `rwkv_ops/delta_net_chunk/torch_triton_kernel.py` | DeltaNet chunkwise PyTorch Triton 桥接 |
| `rwkv_ops/delta_net_chunk/jax_triton_kernel.py` | DeltaNet chunkwise JAX-Triton 桥接 |
| `rwkv_ops/delta_net_chunk_sane/native_keras_op.py` | DeltaNet chunkwise SANE 原生参考实现 |
| `rwkv_ops/delta_net_chunk_sane/triton/` | DeltaNet chunkwise SANE Triton 内核（覆盖 chunk_h / chunk_bwd_dhu，其余复用 delta_net_chunk） |
| `rwkv_ops/delta_net_chunk_sane/torch_triton_kernel.py` | DeltaNet chunkwise SANE PyTorch Triton 桥接 |
| `rwkv_ops/delta_net_chunk_sane/jax_triton_kernel.py` | DeltaNet chunkwise SANE JAX-Triton 桥接 |
| `rwkv_ops/delta_net_recurrent/native_keras_op.py` | DeltaNet recurrent 原生参考实现 |
| `rwkv_ops/delta_net_recurrent/triton_kernel.py` | DeltaNet recurrent 共享 Triton 内核 |
| `rwkv_ops/delta_net_recurrent/torch_triton_kernel.py` | DeltaNet recurrent PyTorch Triton 桥接 |
| `rwkv_ops/delta_net_recurrent/jax_triton_kernel.py` | DeltaNet recurrent JAX-Triton 桥接 |
| `rwkv_ops/delta_net_recurrent/jax_pallas_kernel.py` | DeltaNet recurrent Pallas 内核 |
| `rwkv_ops/delta_net_recurrent/torch_cuda_kernel/` | DeltaNet recurrent PyTorch C++/CUDA 扩展（训练含反向 / 推理 / 单步，按 (K,V,chunk) 懒编译） |
| `rwkv_ops/delta_net_recurrent/jax_cuda_kernel/` | DeltaNet recurrent JAX FFI CUDA（训练含反向 / 推理 / 单步，按 (K,V,chunk) 懒编译） |
| `rwkv_ops/delta_net_recurrent_sane/native_keras_op.py` | DeltaNet recurrent SANE 原生参考实现 |
| `rwkv_ops/delta_net_recurrent_sane/triton_kernel.py` | DeltaNet recurrent SANE 共享 Triton 内核 |
| `rwkv_ops/delta_net_recurrent_sane/torch_triton_kernel.py` | DeltaNet recurrent SANE PyTorch Triton 桥接 |
| `rwkv_ops/delta_net_recurrent_sane/jax_triton_kernel.py` | DeltaNet recurrent SANE JAX-Triton 桥接 |
| `rwkv_ops/delta_net_recurrent_sane/jax_pallas_kernel.py` | DeltaNet recurrent SANE Pallas 内核 |
| `rwkv_ops/delta_net_recurrent_sane/torch_cuda_kernel/` | DeltaNet recurrent SANE PyTorch C++/CUDA 扩展（训练含反向含 dtau / 推理 / 单步） |
| `rwkv_ops/delta_net_recurrent_sane/jax_cuda_kernel/` | DeltaNet recurrent SANE JAX FFI CUDA（训练含反向 / 推理 / 单步，按 (K,V,chunk) 懒编译） |
| `rwkv_ops/rwkv6_kernel/ops_rwkv_kernel.py` | RWKV-6 数值 ground truth |
| `rwkv_ops/rwkv6_kernel/native_keras_op.py` | RWKV-6 函数式原生封装 |
| `rwkv_ops/mhc_kernel/native_op.py` | mHC 原生参考实现 |
| `rwkv_ops/cuda_tools/nvcc_wrap` | nvcc 包装器（绕过 CUDA 13.1/glibc 冲突），勿删、保持可执行 |
| `clean_build_artifacts.py` | 编译产物清理（pytest 会话结束自动调用） |
| `tests/conftest.py` | 共享 fixtures / 断言工具 / 自动清理钩子（不 import 后端） |
| `pyproject.toml` | 包元数据（hatchling）、依赖、版本号 |
| `MANIFEST.in` | 源码分发文件清单 |

---

## 14. DeltaNet（delta_rule，无门控）移植计划

> 来源：flash-linear-attention 的 `fla/ops/delta_rule`。fla 的 delta_rule 与
> gated_delta_rule 共用底层 kernel，唯一语义差别是**没有 `g` 衰减门**
> （decay≡1）。核心递推：
>
> ```text
> kv_mem_t = sum_K(state_{t-1} * k_t)
> delta_t  = (v_t - kv_mem_t) * beta_t
> state_t  = state_{t-1} + k_t ⊗ delta_t
> y_t      = sum_K(state_t * q_t)
> ```
>
> 输出乘 `scale = 1/sqrt(K)`；`q`/`k` 算子内部 L2 norm；`beta` 必须外部已
> 过 sigmoid；state 形状 `[B, H, K, V]`，float32。以上约定与 gdn 完全一致。

### 14.1 命名与 API

新家族对仗 `gated_delta_net_*` → `delta_net_*`：

| 目录 | 公开函数 |
|---|---|
| `rwkv_ops/delta_net_chunk/` | `delta_net_chunk` |
| `rwkv_ops/delta_net_recurrent/` | `delta_net_recurrent` / `delta_net_recurrent_inference` / `delta_net_recurrent_single_step` + 仅 native 的 `delta_net_reference` |
| `rwkv_ops/delta_net_chunk_sane/` | `delta_net_chunk_sane` |
| `rwkv_ops/delta_net_recurrent_sane/` | `delta_net_recurrent_sane` / `..._inference` / `..._single_step` |

签名与 gdn 完全一致、仅去掉 `g` 参数；SANE 版在 `beta` 后插入
`tau [B, T//chunk_size, H]` 与 `mask=None [B, T//chunk_size]`。

### 14.2 分阶段实施与验收

分阶段实施，**不要一次性写完**。每阶段验收流程：测试全绿（分进程跑对应
后端）→ 更新 AGENTS.md / README.md / ENREADME.md 支持矩阵 → git commit 验收
后才进入下一阶段。

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | native 实现（chunk + recurrent 两家族）。重点是测试：`delta_net_chunk` / `delta_net_recurrent` / `delta_net_reference` 三方互拍对齐作为后续加速内核的基准，另做 g≡0 交叉验证（同输入喂 `gated_delta_net_*` 传 `g=zeros` 应一致）。测试覆盖全部五个后端 | 完成 |
| 2 | recurrent Triton（`triton_kernel.py` 共享 kernel + torch/jax 桥接），对照 `gdn_recurrent/` 的实现方式与 API 派生；训练/推理/单步三个算子都要有 Triton 入口 | 完成 |
| 3 | chunk Triton（torch/jax）。chunk 家族**只做 native + Triton**：Pallas 过于复杂不做；CUDA 不做（自研 SIMT gemm 打不过 `tl.dot`） | 完成 |
| 4 | recurrent Pallas（jax）+ CUDA（torch 扩展 / jax FFI），对照阶段 2 的 Triton 逻辑。分发语义：torch 非 CPU 时 native 默认即 Triton；jax GPU/TPU 时 native 默认即 Pallas；cuda 需显式 `KERNEL_TYPE="cuda"` | 完成 |
| 5 | SANE 变体（`delta_net_chunk_sane` / `delta_net_recurrent_sane`）。变化很小（约 95% 代码复用前四阶段产物），全部放最后做 | 完成 |

### 14.3 移植要点

- native 从 `gdn_chunk/native_keras_op.py`、`gdn_recurrent/native_keras_op.py`
  派生：recurrent 删 `state * exp(g_t)` 一行；chunk 删 cumsum、decay 矩阵
  退化为下三角全 1（仍先 mask 上三角）。
- Triton 从 `gdn_recurrent/triton_kernel.py`、`gdn_chunk/triton/` 去 g 派生；
  注意 §9 陷阱 11-13（jax-triton 参数顺序、L2 norm 反向传原始 q/k、
  autotune config 含显式 constexpr 值）。
- recurrent CUDA 是逐步 SIMT 扫描（无 gemm），适合自研，照 `gdn_recurrent`
  的 `torch_cuda_kernel/` / `jax_cuda_kernel/` 派生，按 `(K, V, chunk_size)`
  懒编译；新 FFI 构建目录加进 `clean_build_artifacts._CLEAN_PATTERNS`。
- 测试 fixture：根 `tests/conftest.py` 加 `delta_net_shape` /
  `delta_net_inputs`（`gdn_inputs` 去掉 g）；容差沿用 GDN 惯例。
- 版本号 bump 放到阶段 5 完成后统一做。

---

## 15. 待办：SANE 无 mask 独立算子（进行中）

> **触发规则 A1（已定案，勿改）**：`use_mask = output_final_state and mask is not None`。
> `use_mask=False`（即 `output_final_state=False` 或 `mask=None`）时，chunk 边界**无条件**执行 SANE：
> 不读 mask、不算 `state*(1-m) + sane*m`。native 已如此（`_apply_state_norm_uncond`），加速实现必须对齐。
> **做法（方案 A）**：为每个入口新增**独立的 no-mask kernel**，由 Python 入口按 `use_mask` 二选一；
> no-mask 路径不再分配/传递占位 mask。
> 参照范式：`rwkv7_sane_kernel/triton_kernel.py`（无 mask 前向/反向各一个独立 kernel）与
> `rwkv7_sane_kernel/torch_triton_kernel.py`（`_make_sane_triton_op_with_mask` / `_make_sane_triton_op_no_mask`）。
> 命名：新 kernel 与包装一律加 `_no_mask` 后缀。

### 15.1 范围

| 家族 | 需新增 no-mask 的入口 | 后端 |
|---|---|---|
| `gdn_recurrent_sane` | train-fwd / train-bwd(含 dtau) / inference | Triton(共享+torch+jax)、CUDA(torch 扩展 + jax FFI)、Pallas(jax) |
| `delta_net_recurrent_sane` | 同上 | 同上 |
| `gdn_chunk_sane` | `chunk_h` / `chunk_bwd_dhu`(含 dtau) | 仅 Triton(torch+jax) |
| `delta_net_chunk_sane` | 同上 | 仅 Triton |

**明确不做**：single-step（SANE 由 per-sample `do_sane` 标量控制，无 mask 数组）；native（已有无条件分支）。
**chunk 家族不做 CUDA/Pallas**（设计约定：自研 SIMT 打不过 `tl.dot`）。

### 15.2 已完成

**(a) 已提交 `12484ea dn no mask 算子`**
1. `delta_net_recurrent_sane` 的 jax/cuda 前置修复：`static_argnums`/`nondiff_argnums` 按形参个数重算
   （`jax_triton_kernel.py`、`jax_pallas_kernel.py`、`jax_cuda_kernel/delta_net_recurrent_sane_jax.py`）；
   CUDA `.cu` 与 `.cpp` 的显式实例化列表按声明重新生成（fwd 17 / bwd 22 / inference 13 / single_step 12）。
2. 测试 chunk_size 由 8 改为 32（5 个 `tests/{torch,jax}/test_*_delta_net_recurrent_sane_*`）。
3. chunk 家族 no-mask **内核层**：`gdn_chunk_sane/triton/{chunk_h,chunk_bwd_dhu}.py` 与
   `delta_net_chunk_sane/triton/{chunk_h,chunk_bwd_dhu}.py`，各新增 1 个 `*_no_mask_kernel` + 1 个 `*_no_mask` 包装。

**(b) 工作区未提交（10 个文件，AST OK、ruff 绿）**：chunk 家族 no-mask **桥接层**
- 两个家族的 `triton/__init__.py` 导出 `*_no_mask`；
- `torch_triton_kernel.py`：导入 no-mask kernel、autotune cache 列表、fwd/bwd 按 `use_mask` 二选一分派；
- `jax_triton_kernel.py`：no-mask 分支改用 no-mask 内核，并去掉 `dummy_mask` 与 `USE_MASK=False`。

→ **chunk 家族全链路已通，只差测试**。
