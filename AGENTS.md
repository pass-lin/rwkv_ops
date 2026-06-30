# RWKV OPS 项目 —— 给 AI 协作代理的说明

> 本文件面向自动编程 Agent，补充 README 中未详细描述的工程约定、代码结构与协作规范。
> 项目维护的是 RWKV 系列模型的**核心算子**，不维护 layer 或完整 model。
> 项目依赖 `keras>=3.0`，通过 `KERAS_BACKEND` 统一支持 PyTorch、JAX、TensorFlow、NumPy 多个后端。

---

## 1. 项目定位与总体结构

```text
rwkv_ops/
├── __init__.py              # 包入口：读取环境变量、注册后端、暴露公共 API
├── mhc_kernel/              # mHC (Multi-Head Control) 算子
├── rwkv6_kernel/            # RWKV-6 算子
└── rwkv7_kernel/            # RWKV-7 广义 delta rule 算子

tests/                       # pytest 测试目录（按后端隔离）
├── conftest.py              # 公共 fixtures / 数值对比工具
├── jax/                     # JAX 后端测试
├── numpy/                   # NumPy 后端测试
├── tensorflow/              # TensorFlow 后端测试
└── torch/                   # PyTorch 后端测试

dist/                        # 构建产物（whl）
pyproject.toml               # hatchling 构建配置
```

三个算子家族相互独立，但共享同一套**后端选择机制**和**原生 Keras 参考实现**。

### 1.1 暴露的公共 API

在 `rwkv_ops/__init__.py` 中统一导出：

| 名称 | 说明 |
|---|---|
| `generalized_delta_rule` / `rwkv7_op` | RWKV-7 训练算子（chunkwise） |
| `generalized_delta_rule_inference` / `rwkv7_op_inference` | RWKV-7 推理算子（无梯度） |
| `rnn_generalized_delta_rule` / `rwkv7_op_rnn` | RWKV-7 单步算子（T=1，用于 decode） |
| `rwkv6_op` / `RWKV6_OP` | RWKV-6 函数式算子（`RWKV6_OP` 为兼容别名） |
| `mhc_pre_op` / `mhc_post_op` | mHC 预处理/后处理算子 |
| `get_generalized_delta_rule` | 按 head_size 获取 RWKV-7 算子 |
| `get_rnn_generalized_delta_rule` | 按 head_size 获取 RWKV-7 单步算子 |
| `get_rwkv6_kernel` | 按 head_size/max_sequence_length 获取 RWKV-6 算子 |
| `get_mhc_kernel` | 获取 mHC 算子对 |

---

## 2. 后端选择机制（非常重要）

### 2.1 环境变量

| 变量 | 含义 | 可取值 | 默认值 | 优先级 |
|---|---|---|---|---|
| `KERNEL_BACKEND` | 算子后端 | `jax` / `torch` / `tensorflow` / `numpy` | `torch` | **最高** |
| `KERAS_BACKEND` | Keras 后端 | `jax` / `torch` / `tensorflow` / `numpy` | — | 低 |
| `KERNEL_TYPE` | 实现类型 | `triton` / `cuda` / `native` | `cuda` | — |

选择逻辑（见 `rwkv_ops/__init__.py`）：

1. 若 `KERNEL_BACKEND` 有值，直接使用；
2. 否则若 `KERAS_BACKEND` 有值，使用它；
3. 否则默认 `torch`，并自动 `keras.config.set_backend("torch")`。

`KERNEL_TYPE` 决定具体实现：
- `native`：纯 Keras ops 实现，所有后端都可用，速度最慢，作为 ground truth。
- `cuda`：手写 CUDA 内核（RWKV6/RWKV7）或 CUDA FFI（JAX RWKV7）。
- `triton`：Triton 实现（目前仅 RWKV7 和 mHC 支持）。

### 2.2 各算子的后端支持矩阵

#### RWKV-7 `generalized_delta_rule`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ✅   | ✅     | ✅     |
| JAX       | ✅   | ✅     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |

#### RWKV-7 `rwkv7_op_rnn` (T=1)

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ✅   | ❌     | ✅     |
| JAX       | ✅   | ❌     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |

#### RWKV-6 `rwkv6_op`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ✅   | ❌     | ✅     |
| JAX       | ✅   | ❌     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |

> JAX `cuda` 后端通过 `jax.ffi` 实现，支持 JAX >= 0.4.31（含 0.6.x），
> 当前 CUDA FFI 路径仅对 `bfloat16` I/O 做加速，其它 dtype 自动回退到 `native`。
> ROCm (HIP) 仅 Torch 路径保留；JAX 路径暂不支持 ROCm。

#### mHC `mhc_pre_op` / `mhc_post_op`

| Framework | cuda | triton | native |
|-----------|------|--------|--------|
| PyTorch   | ❌   | ✅     | ✅     |
| JAX       | ❌   | ✅     | ✅     |
| TensorFlow| ❌   | ❌     | ✅     |
| NumPy     | ❌   | ❌     | ✅     |

---

## 3. 代码组织约定

### 3.1 每个算子家族的目录结构模式

以 `rwkv7_kernel` 为例：

```text
rwkv7_kernel/
├── __init__.py                    # 工厂函数，根据后端+KERNEL_TYPE 选择实现
├── native_keras_op.py             # 纯 Keras 参考实现（ground truth）
├── triton_kernel.py               # 共享 Triton 内核（被 JAX/Torch 共用）
├── jax_triton_kernel.py           # JAX ↔ Triton 桥接（custom_vjp + SPMD）
├── torch_triton_kernel.py         # PyTorch ↔ Triton 桥接（autograd.Function）
├── jax_cuda_kernel/               # JAX FFI CUDA
│   ├── wkv7_jax.py
│   ├── wkv7_ffi.cu
│   └── CMakeLists.txt
├── jax_cuda_kernel_single/        # JAX FFI CUDA 单步
├── torch_cuda_kernel/             # PyTorch C++/CUDA 扩展
│   ├── wkv7_torch.py
│   ├── wkv7_cuda.cu
│   └── wkv7_op.cpp
└── torch_cuda_kernel_single/      # PyTorch C++/CUDA 单步
```

```text
rwkv6_kernel/
├── __init__.py                    # 工厂函数，按后端+KERNEL_TYPE 选择实现
├── native_keras_op.py             # 纯 Keras 参考实现（ground truth）
├── ops_rwkv_kernel.py             # RWKV-6 原生参考实现（保留的数值基准）
├── jax_cuda_kernel/               # JAX FFI CUDA
│   ├── wkv6_jax.py
│   ├── wkv6_ffi.cu
│   └── CMakeLists.txt
└── torch_cuda_kernel/             # PyTorch C++/CUDA 扩展
    ├── wkv6_torch.py
    ├── wkv6_cuda.cu
    └── wkv6_op.cpp
```

RWKV-6 的 CUDA 核心计算逻辑复用同一套 kernel 模板；JAX FFI 与 Torch C++ 扩展分别为其提供 XLA FFI 与 `torch.autograd.Function` 桥接。

### 3.2 原生实现的地位

- `native_keras_op.py` / `ops_rwkv_kernel.py` / `native_op.py` 是**数值对齐的基准**。
- 新增 CUDA/Triton 实现时，必须保证与原生实现的数值一致性。
- 原生实现使用 `keras.ops` 编写，因此天然跨后端。

### 3.3 精度约定

- 原生实现内部大量使用 `float32` 计算，最后 `cast` 回输入 dtype。
- CUDA/Triton 内核通常使用 `bfloat16` I/O，`float32` 内部累加。
- RWKV6 的 fp16 输入会输出 fp32；fp32/bf16 保持同类型。
- mHC Triton 内核的 `H_post` / `H_res` 输出固定为 `float32`。

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

- 状态 `state` 形状：`(B, H, K, K)`，其中 `H` 是 head 数，`K` 是 head_size。
- 输入默认 layout：`[B, T, H, K]`；`head_first=True` 时内部会转置为 `[B, H, T, K]`。
- `mask` 形状 `(B, T)`，`1` 表示更新状态，`0` 表示冻结状态。
- **注意**：chunkwise 内核要求 `T % 16 == 0`。

### 4.2 RWKV-7 的 `head_size` 处理

- `triton` 后端：要求 `HEAD_SIZE % 4 == 0`，当前 Triton 实现固定支持 `HEAD_SIZE == 64`（Torch 路径有显式检查）。
- `cuda` 后端：`head_size` 通过 `-D_C_` 编译进内核，首次使用会按 head_size 懒编译。
- 若 `head_size != 64`，应使用 `get_generalized_delta_rule(your_head_size, KERNEL_TYPE="cuda")`。

### 4.3 RWKV-6

- `rwkv6_op` 为**函数式接口**，无需实例化：
  ```python
  y, final_state = rwkv6_op(
      r, k, v, w, u,
      initial_state=None,
      output_final_state=False,
      state_map=None,
      head_first=False,
  )
  ```
- `max_sequence_length` 是 CUDA kernel 的**编译期常量**，不同 `(head_size, T)` 组合会各自懒编译一个 shared object。
- 输入 layout：`[B, T, C]`（`head_first=False`）或 `[B, H, T, N]`（`head_first=True`），其中 `C = H * N`。
- `u` 形状：`(H, N)` 或 `(C,)`。

### 4.4 mHC

- Pre-Op：多流 `x [B, T, n, C]` → 单流 `x_layer_in [B, T, C]` + `H_post [B, T, n]` + `H_res [B, T, n, n]`。
- Post-Op：单流 `layer_out` + 原多流 `x` + `H_post` + `H_res` → 新多流 `x_next`。
- 内部流程：`linear_and_reshape` → `Sinkhorn-Knopp`（生成双随机矩阵）→ `stream_aggregate`。
- **约束**：`C` 必须能被 128 整除；投影矩阵输出维度 `M = n*(n+2)` 必须是 32 的倍数。

---

## 5. 构建与编译

### 5.1 包构建

- 使用 `hatchling` 作为构建后端（见 `pyproject.toml`）。
- `pyproject.toml` 中 `dependencies = ["keras>=3.0"]`。
- `MANIFEST.in` 控制源码分发内容：包含所有 `.py`、CUDA/HIP/C++ 源文件、CMake 文件；排除 `build/`、`dist/`、编译后的 `.so` 等。
- 版本号需同时维护 `pyproject.toml` 和 `rwkv_ops/__init__.py` 中的 `__version__`。

### 5.2 CUDA 扩展的懒编译

- RWKV6/7 的 CUDA 实现不在 whl 中预编译，而是**首次 import 时通过 CMake（JAX FFI）或 `torch.utils.cpp_extension.load`（Torch）动态编译**。
- 编译产物会写入源码树下的 `build_<head_size>[_<max_sequence_length>]` 或 `builds/` 目录，并被 `.gitignore` 排除。
- 因此：
  - 不要提交 `.so`、`.o`、`build*` 等编译产物；
  - 修改 C++/CUDA 源码后，需要手动删除对应 `build*` 目录才能触发重新编译。

### 5.3 运行环境要求

- CUDA 路径：JAX 编译依赖 `/usr/local/cuda` 软链接，如不存在需手动创建。
- GCC/NVCC 兼容性：CMake 调用的 host C++ 编译器（`gcc/g++`）必须与 CUDA 版本兼容。若系统 GCC 过新（如 GCC 15 + CUDA 13.1），可在运行前通过环境变量指定兼容版本：
  ```bash
  export CC=/path/to/gcc-13
  export CXX=/path/to/g++-13
  export CUDAHOSTCXX=/path/to/g++-13
  ```
- ROCm：RWKV6 Torch 路径支持 HIP，通过 `RWKV_USE_ROCM=1` 环境变量启用；JAX FFI 路径暂不支持 ROCm。
- PyTorch CUDA 扩展依赖 `ninja`。

---

## 6. 测试规范

### 6.1 测试入口

项目使用 pytest，按后端分目录隔离（每个目录的 `conftest.py` 在导入 keras 前设置 `KERAS_BACKEND`）：

```bash
# 安装测试依赖
pip install -e ".[test]"

# 各后端入口
pytest tests/torch -v
pytest tests/jax -v
pytest tests/numpy -v
pytest tests/tensorflow -v

# 跳过较重的 slow 测试
pytest tests/torch tests/jax -v -m "not slow"
```

> **注意**：同一 Python 进程内 Keras 后端只能设定一次，因此 torch/jax 测试必须分进程运行，不要在一个 pytest 进程里同时跑 `tests/torch` 和 `tests/jax`。
>
> **文件名约定**：不同后端目录下的测试文件必须保持**唯一的模块名**（例如 `test_torch_rwkv6.py` / `test_jax_rwkv6.py`）。如果多个目录存在同名 `test_rwkv6.py`，pytest 在顶层收集时会发生 `import file mismatch`，导致部分后端测试无法被发现。
>
> **JAX 编译器自动选择**：`tests/jax/conftest.py` 会在导入前检测系统 GCC 版本。若默认 GCC 版本过高（如 GCC 15 + CUDA 13.1），会自动在 `PATH` 中寻找 `gcc-13`/`g++-13`、`gcc-12`/`g++-12` 或 `x86_64-conda-linux-gnu-gcc/g++` 并设置 `CC`/`CXX`/`CUDAHOSTCXX`；若用户已显式设置这些变量，则保持用户配置不变。

### 6.2 测试覆盖要求

- 新增 CUDA/Triton 内核时，必须在对应后端添加数值对比测试：
  - 前向输出与 `native` 实现对比；
  - 反向梯度与 `native` 实现或自动微分对比；
  - mask 全 1、全 0、随机 mask 的等价性检查；
  - `head_first=True/False` 两种 layout。
- mHC 测试同时包含速度和显存基准。

### 6.3 运行单个测试

```bash
# RWKV6 Torch CUDA vs native
pytest tests/torch/test_torch_rwkv6.py -v

# RWKV7 JAX CUDA
pytest tests/jax/test_jax_rwkv7.py -v

# mHC Triton（Torch）
pytest tests/torch/test_torch_mhc_post_op.py -v

# 原生 NumPy smoke
pytest tests/numpy/test_numpy_native_simple.py -v
```

---

## 7. 代码风格与协作规范

### 7.1 Python 代码

- 使用 `keras.ops` 编写后端无关逻辑，避免直接调用 `torch.*` / `jax.*` / `tf.*`，除非在特定后端绑定文件中。
- 所有公开函数/类需要中文或英文 docstring，说明：
  - 输入形状；
  - 输出形状；
  - 重要约束（如 `T % 16 == 0`、`C % 128 == 0`）。
- 数值计算优先在 `float32` 下进行，最后 `cast` 回输入 dtype。

### 7.2 CUDA/Triton 代码

- 编译期常量通过宏传入：`-D_N_=64 -D_T_=4096`。
- CUDA 指针算术使用 64 位整数，防止大 tensor 溢出。
- Triton 内核使用 `tl.int64` 进行指针运算。
- 新增 Triton 内核时，JAX 和 PyTorch 的桥接文件需要同时更新，并共享同一套 `triton_kernel.py`。

### 7.3 提交前检查清单

- [ ] 版本号是否需要同步更新？
- [ ] 是否新增了编译产物需要被 `.gitignore` 排除？
- [ ] 是否新增了需要被 `MANIFEST.in` 包含的源文件？
- [ ] 新增/修改的算子是否在 `rwkv_ops/__init__.py` 中正确暴露？
- [ ] 是否补充了对应后端的测试？
- [ ] 是否验证了与 `native` 实现的数值一致性？

---

## 8. 常见陷阱

1. **JAX RWKV6 `cuda` 后端已迁移到 `jax.ffi`**，需要 JAX >= 0.4.31；旧版 XLA custom-call 代码已移除。
2. **RWKV7 Triton 后端目前主要验证 HEAD_SIZE=64**，其他 head_size 请用 `cuda` 后端。
3. **chunkwise RWKV7 要求序列长度能被 16 整除**，否则可能静默出错或触发未定义行为。
4. **RWKV6 的 `max_sequence_length` 是编译期常量**，修改后必须删除旧 build 目录重新编译。
5. **mHC 的 `C` 必须被 128 整除**，否则 Triton 内核会报错。
6. **不要直接修改 `dist/` 或 `build/` 目录中的内容**，这些由构建流程生成。

---

## 9. 扩展指南

若新增一种算子：

1. 在 `rwkv_ops/` 下新建目录，至少包含 `__init__.py` 和 `native_*.py`。
2. 在 `__init__.py` 中实现工厂函数，按 `KERNEL_TYPE` + `keras.config.backend()` 分发。
3. 为每个加速后端编写桥接文件，共享同一套 Triton/CUDA 内核。
4. 在 `rwkv_ops/__init__.py` 中导入并暴露。
5. 在 `tests/<backend>/` 中添加对应后端的 pytest 测试；文件名需与已有后端保持唯一模块名，避免 `import file mismatch`。
6. 更新本 `AGENTS.md`、README.md、ENREADME.md 中的支持矩阵。

---

## 10. 相关文件速查

| 文件 | 作用 |
|---|---|
| `rwkv_ops/__init__.py` | 包入口、环境变量解析、API 暴露 |
| `rwkv_ops/rwkv7_kernel/__init__.py` | RWKV7 后端分发器 |
| `rwkv_ops/rwkv7_kernel/native_keras_op.py` | RWKV7 原生参考实现 |
| `rwkv_ops/rwkv7_kernel/triton_kernel.py` | RWKV7 共享 Triton 内核 |
| `rwkv_ops/rwkv6_kernel/__init__.py` | RWKV6 后端分发器 |
| `rwkv_ops/rwkv6_kernel/native_keras_op.py` | RWKV6 函数式原生封装 |
| `rwkv_ops/rwkv6_kernel/ops_rwkv_kernel.py` | RWKV6 原生参考实现（ground truth） |
| `rwkv_ops/rwkv6_kernel/jax_cuda_kernel/wkv6_jax.py` | RWKV6 JAX FFI 桥接 |
| `rwkv_ops/rwkv6_kernel/torch_cuda_kernel/wkv6_torch.py` | RWKV6 Torch C++ 扩展桥接 |
| `rwkv_ops/mhc_kernel/__init__.py` | mHC 后端分发器 |
| `rwkv_ops/mhc_kernel/native_op.py` | mHC 原生参考实现 |
| `pyproject.toml` | 包元数据、构建配置、依赖 |
| `MANIFEST.in` | 源码分发文件清单 |
| `test_*.sh` | 测试套件入口 |
