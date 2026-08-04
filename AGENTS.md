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
├── mhc_kernel/              # mHC (Multi-Head Control) 算子
├── rwkv6_kernel/            # RWKV-6 算子
├── rwkv7_kernel/            # RWKV-7 广义 delta rule 算子
└── rwkv7_sn_kernel/         # RWKV-7 State Neutralization 算子

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

四个算子家族相互独立，但共享同一套**后端选择机制**和**原生 Keras 参考实现**。

### 1.1 暴露的公共 API

在 `rwkv_ops/__init__.py` 中统一导出（import 即按环境变量实例化）：

| 名称 | 说明 |
|---|---|
| `generalized_delta_rule` / `rwkv7_op` | RWKV-7 训练算子（chunkwise） |
| `generalized_delta_rule_inference` / `rwkv7_op_inference` | RWKV-7 推理算子（无梯度，省显存） |
| `rnn_generalized_delta_rule` / `rwkv7_op_rnn` | RWKV-7 单步算子（T=1，用于 decode） |
| `generalized_delta_rule_sn` / `rwkv7_op_sn` | RWKV-7-SN 训练算子 |
| `generalized_delta_rule_sn_inference` / `rwkv7_op_sn_inference` | RWKV-7-SN 推理算子（T 不必被 16 整除） |
| `rnn_generalized_delta_rule_sn` / `rwkv7_op_sn_rnn` | RWKV-7-SN 单步算子 |
| `rwkv6_op` / `RWKV6_OP` | RWKV-6 函数式算子（`RWKV6_OP` 为兼容别名） |
| `mhc_pre_op` / `mhc_post_op` | mHC 预处理/后处理算子 |
| `get_generalized_delta_rule` 等 6 个工厂函数 | 按 head_size / KERNEL_TYPE 获取算子 |

---

## 2. 后端选择机制（非常重要）

### 2.1 环境变量

| 变量 | 含义 | 可取值 | 默认值 | 优先级 |
|---|---|---|---|---|
| `KERNEL_BACKEND` | 算子后端 | `jax` / `torch` / `tensorflow` / `numpy` / `openvino` | `torch` | **最高** |
| `KERAS_BACKEND` | Keras 后端 | `jax` / `torch` / `tensorflow` / `numpy` / `openvino` | — | 低 |
| `KERNEL_TYPE` | 实现类型 | `triton` / `cuda` / `native` / `pallas` | `cuda` | — |
| `RWKV_OPS_PALLAS_AUTOTUNE` | Pallas autotune 开关 | `1` / `0` | `1` | — |

选择逻辑（见 `rwkv_ops/__init__.py`）：

1. 若 `KERNEL_BACKEND` 有值，直接使用；
2. 否则若 `KERAS_BACKEND` 有值，使用它；
3. 否则默认 `torch`，并自动 `keras.config.set_backend("torch")`。

`KERNEL_TYPE` 决定具体实现：

- `native`：纯 Keras ops 实现，所有后端可用，速度最慢，作为 ground truth。
- `cuda`：手写 CUDA kernel（Torch C++ 扩展 / JAX FFI）。
- `triton`：Triton 实现（rwkv7 / rwkv7_sn / mhc）。
- `pallas`：Pallas 实现（仅 JAX 的 rwkv7 / rwkv7_sn）。

**Pallas 默认规则**：`jax` 后端 + `KERNEL_TYPE ∈ {native, pallas}` + 平台为
GPU/TPU 时，rwkv7/rwkv7_sn 默认使用 `jax_pallas_kernel.py`；CPU 回落 native。
单步 RNN 没有 pallas 版本。

**缺硬件静默回退**：各工厂在硬件/库不可用时不报错，直接回退 native
（例如 torch 无 CUDA、jax 不在 GPU/TPU 上）。

### 2.2 各算子的后端支持矩阵

#### RWKV-7 `generalized_delta_rule`

| Framework | cuda | triton | native | pallas |
|-----------|------|--------|--------|--------|
| PyTorch   | ✅   | ✅     | ✅     | ❌     |
| JAX       | ✅   | ✅     | ✅     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     | ❌     |
| NumPy     | ❌   | ❌     | ✅     | ❌     |
| OpenVINO  | ❌   | ❌     | ✅     | ❌     |

#### RWKV-7-SN `generalized_delta_rule_sn`

同 rwkv7op（PyTorch/JAX 全后端 ✅，其余仅 native）。

#### RWKV-7 `rwkv7_op_rnn` / SN `rwkv7_op_sn_rnn` (T=1)

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
| PyTorch   | ❌   | ✅     | ✅     |
| JAX       | ❌   | ✅     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |
| OpenVINO  | ❌   | ❌     | ✅     |

---

## 3. 代码组织约定

### 3.1 每个算子家族的目录结构模式

以 `rwkv7_kernel` 为例（`rwkv7_sn_kernel` 完全平行）：

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
- chunkwise 内核要求 `T % 16 == 0`（cuda 训练版 / triton / pallas）。
- `get_generalized_delta_rule` 返回 `(训练算子, 推理算子)`；推理版不存 checkpoint。
- `triton` 后端固定支持 `HEAD_SIZE == 64`；其他 head_size 用 `cuda`
  （`head_size` 经 `-D_C_` 编译进内核，按 head_size 懒编译）。
- 单步 `get_rnn_generalized_delta_rule` 只支持 cuda，其余 KERNEL_TYPE 回退 native。

### 4.2 RWKV-7 State Neutralization（`rwkv7_sn_kernel`）

在 RWKV-7 递推之上，于 chunk 边界（每 16 tokens）做 State Neutralization：

```text
sn_state = tau * tanh(state / tau)      # 软裁剪到 [-tau, tau]
```

- **tau 语义**：只接收预处理后的 `tau = softplus(param) + 1.0`，严格 > 1，
  形状 `[B, T//16, H]`（per-head per-chunk）。
- **mask 语义**：`[B, T//16]` per-chunk，所有 head 共享。mask=1 在 chunk 边界
  执行 SN；mask=0 保留原 state。不再用 `tau=0.0` 兼任 mask；全 padding chunk
  的 mask 须置 0。kernel 用 `mask * sn_state + (1 - mask) * state` 的 blend 形式，
  避免 warp 分支。
- **padding 处理**：padding 位仍需保证 `k=0, a=0, w=-inf`，且对应 chunk mask=0。
- **输出与 State 的关系**：输出始终基于 SN **之前**的 State；SN 只修改传递给
  下一步/下一 chunk 的 State。训练 kernel 的 checkpoint 保存 **SN 之前** 的
  State 供反向使用；反向先算 `dtau`，再把 `dstate`/`dstateT` 乘 `sech2`。
- **无 mask 算子**：`output_final_state=False` 或 `mask=None` 时调用独立
  no-mask kernel（chunk 边界无条件 SN，不读 mask、不算 blend）。
  `mask=None` 且 `output_final_state=True` 时 Python 入口发双语 UserWarning
  并把 `final_state` 置为 `None`（避免误用被 padding 污染的 state）。
- **推理 kernel**：`generalized_delta_rule_sn_inference` 只输出 y 与最终 state，
  显存显著低于训练版；按 chunk 读 tau，**T 不要求被 16 整除**。任意长度
  prefill 也可用单步 `generalized_delta_rule_sn_single_step`（每步算 SN，
  按 per-sample `do_sn` 选择）。
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
  `fori_loop` 遍历 chunk、内层静态展开 16 步；数学与 `triton_kernel.py`
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

## 5. 构建与编译

### 5.1 包构建

- 构建后端 **hatchling**；运行时唯一依赖 `keras>=3.0`；测试可选依赖
  `pip install -e ".[test]"`（`pytest>=8.0`、`jax-triton>=0.3.1`）。
- **wheel 为纯 Python**（`py3-none-any`），CUDA/Triton 全部运行时 JIT 编译，
  发布物不含二进制 kernel。构建：`python -m build`。
- **版本号两处同步**：`pyproject.toml` 与 `rwkv_ops/__init__.py` 的 `__version__`。
- `MANIFEST.in` 包含所有 `.py`、CUDA/HIP/C++ 源、CMake 文件；排除 `build*/`、
  `dist/`、`.so` 等。新增编译产物类型要同步加排除规则；**不要手改 `dist/`**。

### 5.2 CUDA 扩展的懒编译

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

### 5.3 运行环境要求

- CMake 的 host C++ 编译器须与 CUDA 兼容；GCC 过新（如 GCC 15 + CUDA 13.1）时
  指定 `CC`/`CXX`/`CUDAHOSTCXX` 到 gcc-13 等（`tests/jax/conftest.py` 会自动探测）。
- ROCm：仅 RWKV-6 Torch 路径，`RWKV_USE_ROCM=1`。
- PyTorch CUDA 扩展依赖 `ninja`。

### 5.4 构建产物清理

- `clean_build_artifacts.clean_all()` 删除各 `jax_cuda_kernel*/build_*` 与
  `*.so`、ninja 日志、全部 `__pycache__`。
- `tests/conftest.py` 的 `pytest_sessionfinish` 自动调用——**每次跑完 pytest，
  JAX CUDA 测试下次会重新编译**（耗时是预期行为）。
- 新增带 FFI 的算子目录时，把它的 `build_*`/`*.so` 路径加进 `_CLEAN_PATTERNS`。

---

## 6. 测试规范

### 6.1 测试入口

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

### 6.2 共享 fixtures 与输入分布（`tests/conftest.py`）

- 根 conftest **刻意不 import torch/jax/keras/rwkv_ops**，避免收集阶段锁定后端。
- `rng` 固定种子 42；`rwkv7_shape = (5, 128, 6, 64)`（B, T, H, K）。
- `rwkv7_inputs`：`r/k/v ~ N(0,1)`；`a`、`b` 为同一 z 的 ±单位向量（b = -a）；
  `w = -softplus(w_raw) - 0.5`；`h0 ~ N(0,1)`。
- `rwkv7_sn_inputs`：加 `tau`，`x ~ N(7.0, 0.5)`、`tau = softplus(x) + 1.0`
  （tau ≈ 1000，近似恒等映射），形状 `[B, T//16, H]`。
- **递推数值对拍必须用这些 fixture 的稳定分布**；随手造的随机数据会让
  delta-rule 递推指数发散，f32 参考自身误差都能到 1e5，无法用于判定。
- 断言工具 `assert_allclose_with_stats`：打印 exact/close/max/mean diff，
  判定只看 atol/rtol。

### 6.3 测试覆盖要求与容差惯例

- 新增/修改加速内核必须覆盖：前向输出与 final_state vs native；反向梯度
  vs native 自动微分；mask 全 1 / 全 0 / 随机 mask 等价性；
  `head_first=True/False` 两种 layout。
- SN 类算子额外镜像覆盖：带/不带 mask 的前向+反向（含 tau 梯度）、无 mask
  警告 + None state、任意长度推理（T=34）、不规则 padding、head 轴 TP（jax）。
- 统一模式：同一组 numpy 输入分别喂 native 与加速算子，输入 cast bf16
  （state/tau/mask 保持 f32），loss 用 `mean(y²) + mean(state²)`。
- 容差惯例：
  - RWKV-7 前向 y atol=1e-5 / rtol=1e-2（SN 的 y 放宽到 atol=1e-4）；
    final_state atol=1e-5 / rtol=1e-3；反向 grad atol=7e-3 / rtol=1e-3
    （grad_b 放宽到 1e-2）。
  - RWKV-6 / mHC：一律 1e-2 / 1e-2。
- mHC 测试同时包含速度和显存基准。

---

## 7. 代码风格与协作规范

### 7.1 Python 代码

- `ruff check .` 与 `ruff format --check .` 必须全绿（ruff 无规则定制，
  下列约定靠 AGENTS.md 约束）。
- 禁止 `from ... import *`；禁止 lambda 赋值（`grid = lambda ...`），改用 `def`。
- 函数式入口：每个 kernel 的 Python 入口返回 `out` 或 `(out, final_state)`。
- 使用 `keras.ops` 编写后端无关逻辑；`torch.*` / `jax.*` / `tf.*` 只允许出现在
  对应后端绑定文件中。
- 所有公开函数/类需要中文或英文 docstring，说明输入形状、输出形状、重要约束
  （如 `T % 16 == 0`、`C % 128 == 0`）。

### 7.2 CUDA/Triton/Pallas 代码

- 编译期常量通过宏传入：`-D_N_=64 -D_T_=4096`。
- CUDA 指针算术使用 64 位整数；Triton 内核使用 `tl.int64`。
- 新增 Triton 内核时，JAX 和 PyTorch 的桥接文件需要同时更新，并共享同一套
  `triton_kernel.py`。
- Pallas 内核只用公开 API（见 §4.5）。

### 7.3 提交前检查清单

- [ ] `ruff check .` 与 `ruff format --check .` 全绿。
- [ ] 版本号两处同步（`pyproject.toml` + `rwkv_ops/__init__.py`）。
- [ ] 新增编译产物已加入 `.gitignore` 排除 / `MANIFEST.in` 包含规则。
- [ ] 新增/修改的算子已在 `rwkv_ops/__init__.py` 正确暴露。
- [ ] 对应后端测试已补充（文件名带后端前缀），并验证与 native 数值一致。
- [ ] Triton kernel 改动已同步检查 jax/torch 两个桥接。
- [ ] 新 FFI 算子的构建目录已加入 `clean_build_artifacts._CLEAN_PATTERNS`。
- [ ] 已更新 `AGENTS.md`、`README.md`、`ENREADME.md` 的支持矩阵。

---

## 8. 常见陷阱

1. **JAX RWKV6 `cuda` 后端用 `jax.ffi`**，需要 JAX >= 0.4.31；旧版 XLA
   custom-call 代码已移除。
2. **RWKV7 Triton 后端目前主要验证 HEAD_SIZE=64**，其他 head_size 请用 `cuda`。
3. **chunkwise RWKV7 要求序列长度能被 16 整除**，否则可能静默出错或触发
   未定义行为。
4. **RWKV6 的 `max_sequence_length` 是编译期常量**，修改后必须删除旧 build
   目录重新编译。
5. **mHC 的 `C` 必须被 128 整除**（Triton 入口已加显式校验）。
6. **不要直接修改 `dist/` 或 `build/` 目录中的内容**。
7. **`custom_partitioning` 在 eager 下也会 trace 内层函数**：任何运行期探测
   （如 pallas autotune）必须在 trace 之外完成（见 §4.5 warmup 约定）。
8. **JAX CUDA 测试每次会话后 `.so` 被自动清理**，下次重编译是预期行为。
9. **rwkv6 的 `KERNEL_TYPE="triton"` 静默回退 native**，不报错。
10. **递推对拍必须用 fixture 的稳定输入分布**（见 §6.2），随手造的随机数据
    会让递推指数发散，得出"实现错了"的假结论。

---

## 9. 扩展指南

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

## 10. 相关文件速查

| 文件 | 作用 |
|---|---|
| `rwkv_ops/__init__.py` | 包入口、环境变量解析、API 暴露（import 即实例化全部算子） |
| `rwkv_ops/pallas_utils.py` | Pallas 后端公共机制：候选配置、autotune、SPMD 辅助 |
| `rwkv_ops/rwkv7_kernel/__init__.py` | RWKV-7 后端分发器 |
| `rwkv_ops/rwkv7_kernel/native_keras_op.py` | RWKV-7 原生参考实现 |
| `rwkv_ops/rwkv7_kernel/triton_kernel.py` | RWKV-7 共享 Triton 内核 |
| `rwkv_ops/rwkv7_kernel/jax_pallas_kernel.py` | RWKV-7 Pallas 内核 |
| `rwkv_ops/rwkv7_sn_kernel/` | SN 版，结构与 rwkv7_kernel 完全平行 |
| `rwkv_ops/rwkv6_kernel/ops_rwkv_kernel.py` | RWKV-6 数值 ground truth |
| `rwkv_ops/rwkv6_kernel/native_keras_op.py` | RWKV-6 函数式原生封装 |
| `rwkv_ops/mhc_kernel/native_op.py` | mHC 原生参考实现 |
| `rwkv_ops/cuda_tools/nvcc_wrap` | nvcc 包装器（绕过 CUDA 13.1/glibc 冲突），勿删、保持可执行 |
| `clean_build_artifacts.py` | 编译产物清理（pytest 会话结束自动调用） |
| `tests/conftest.py` | 共享 fixtures / 断言工具 / 自动清理钩子（不 import 后端） |
| `pyproject.toml` | 包元数据（hatchling）、依赖、版本号 |
| `MANIFEST.in` | 源码分发文件清单 |
