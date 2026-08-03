# Agent Notes for rwkv_ops

RWKV 系列 kernel 的多后端 Keras 算子库。算子：RWKV-7（generalized delta rule）、
RWKV-7-SN（State Neutralization）、RWKV-6、mHC。每个算子有 native（Keras ops，
ground truth）+ 若干加速后端（CUDA / Triton / Pallas）。

## 1. 环境与分发

- 环境变量（`rwkv_ops/__init__.py` 在 import 时解析并实例化全部算子）：
  - `KERNEL_TYPE`：`cuda`（默认）/ `triton` / `native` / `pallas`。
  - `KERAS_BACKEND` 优先于 `KERNEL_BACKEND`；都未设时强制 `torch`。
  - `RWKV_OPS_PALLAS_AUTOTUNE`：`1`（默认）/ `0`，Pallas 后端 autotune 开关。
- 每个算子目录的 `__init__.py` 是工厂：`get_xxx(HEAD_SIZE=64, KERNEL_TYPE=...)`，
  按 `keras.config.backend()` × `KERNEL_TYPE` × 硬件可用性分发；**缺硬件时静默
  回退 native，不报错**。
- Pallas 默认规则：`jax` 后端 + `KERNEL_TYPE ∈ {native, pallas}` + 平台
  GPU/TPU → 使用 `jax_pallas_kernel.py`；CPU 回落 native。单步 RNN 无 pallas 版。
- 工厂返回二元组 `(训练算子, 推理算子)`（rwkv6 与 mhc 除外，见各自小节）。

## 2. 代码风格

- `ruff check .` 与 `ruff format --check .` 必须全绿（ruff 无规则定制，下列约定靠自觉）：
  - 禁止 `from ... import *`；禁止 lambda 赋值（`grid = lambda ...`），改用 `def`。
  - 函数式入口：每个 kernel 的 Python 入口返回 `out` 或 `(out, final_state)`，无类状态。
- 后端无关逻辑用 `keras.ops`；`torch.*`/`jax.*` 只允许出现在对应后端的绑定文件中。
- 公开函数需中/英 docstring，写明输入/输出形状与重要约束（如 `T % 16 == 0`）。
- 数值计算优先 float32，最后 cast 回输入 dtype；CUDA kernel 输入强制 bf16，
  State/tau/中间量保持 fp32；dtype 不符时**警告并 cast，不报错**。

## 3. 跨算子共享模式

- **共享 Triton kernel**：`triton_kernel.py`（rwkv7/sn）或 `triton_kernel/` 子目录
  （mhc）同时被 `jax_triton_kernel.py` 与 `torch_triton_kernel.py` import。
  **改 Triton kernel 必须同步检查 jax/torch 两个桥接。**
- **JAX SPMD 三件套**：`custom_partitioning` + `def_partition(
  infer_sharding_from_operands, sharding_rule, partition)` + `_create_partition`
  样板。sharding rule 用同一字母标记所有 head 维度以支持 head 轴 TP。
- **命名规律**：训练 chunkwise 在 `jax_cuda_kernel/`、`torch_cuda_kernel/`；
  单步 RNN 在 `*_single/` 目录、文件带 `_single_step` 后缀。

## 4. 各后端实现要点

### CUDA

- 编译期常量经宏传入（`-D_N_=64 -D_CHUNK_LEN_=16` 等）；改 HEAD_SIZE 会生成新的
  `build_<...>/` 目录。
- 指针算术用 64 位整数（防大 tensor 溢出）。
- JAX 桥接：`cmake` + `jax.ffi.register_ffi_target`（需 JAX ≥ 0.4.31），首次调用时
  编译 `build_<...>/wkv*.so`，经 `ctypes.CDLL` 加载；编译器一律注入
  `rwkv_ops/cuda_tools/nvcc_wrap`（用 `include/bits/mathcalls.h` 的 `#include_next`
  绕过 CUDA 13.1 与 glibc 2.41+ 的 `rsqrt/rsqrtf` 冲突）。**不要修改系统 CUDA 头
  文件**；`nvcc_wrap` 需保持可执行权限。
- Torch 桥接：`torch.utils.cpp_extension.load` 延迟编译

### Triton

- 指针运算用 `tl.int64`；`MINI_BSZ=1`（每程序一个 batch-head，规避 ROCm 寄存器
  溢出）。
- rwkv7/sn 的 Triton 后端主要验证 `HEAD_SIZE == 64` 且要求 `T % 16 == 0`。
- ROCm通过triton来做兼容，其他后端不考虑triton

### Pallas（`jax_pallas_kernel.py`，公共机制在 `rwkv_ops/pallas_utils.py`）

- kernel 本体**只用公开稳定 Pallas API**：`pl.pallas_call` / `pl.BlockSpec` /
  `pl.program_id` / ref 索引 / `lax.fori_loop` / `jnp`。**禁止 import
  plgpu/pltriton 私有 API**，以兼容老版 jax 的 Triton lowering、未来版本的
  Mosaic GPU lowering 以及 TPU。
- 结构：grid=(B, N) 每程序一个 (batch, head)；外层 `fori_loop` 遍历 chunk，
  内层静态展开 16 步；数学与 `triton_kernel.py` 逐行对应。
- autotune（pallas_utils）：候选 = triton `num_warps×num_stages` 网格 + mgpu，
  eager 下对新 shape 计时选最快，**不可编译的候选自动跳过**（这也是后端能力
  探测：jax 0.10.x 的 MGPU lowering 对逐行动态索引有 128 元素向量约束，会被
  自动跳过而落回 triton lowering）。
- **warmup 约定**：`custom_partitioning` eager 调用也会 trace 内层函数，因此
  每个 custom_vjp 入口（primal/`_fwd`/`_bwd`）必须先用真实数组调 warmup
  （`ensure_config`）解析配置；被 trace 的路径只查缓存，取不到用首个候选。
  **warmup 的参数顺序必须与 launcher 一致**（曾有因此导致的隐蔽编译失败）。
- 新增 pallas kernel 必须复用 `pallas_utils`，不要内联复制 autotune/SPMD 机制。

## 5. 算子族约定

### RWKV-7（`rwkv7_kernel`）

- mask 为 **per-token** `[B, T]`（或 `[B, T, 1, 1]`），mask=0 处 state 冻结。
- chunkwise 要求 `T % 16 == 0`（triton/pallas/cuda 训练版）。
- `get_generalized_delta_rule` 返回 `(训练, 推理)`；推理版不存 checkpoint 省显存。
- 单步 `get_rnn_generalized_delta_rule` 只支持 cuda，其余回退 native。

### RWKV-7 State Neutralization（`rwkv7_sn_kernel`）

- **tau 语义**：只接收预处理后的 `tau = softplus(param) + 1.0`，严格 > 1，
  形状 `[B, T//16, H]`（per-head per-chunk）。
- **mask 语义**：`[B, T//16]` per-chunk，所有 head 共享。mask=1 在 chunk 边界
  执行 `state = tau * tanh(state / tau)`；mask=0 保留原 state。不再用 tau=0 兼任
  mask；全 padding chunk 的 mask 须置 0。
- **padding**：padding 位仍需 `k=0, a=0, w=-inf`，且对应 chunk mask=0。
- **输出与 State**：输出始终基于 SN **之前**的 State；训练 kernel 的 checkpoint
  保存 SN 之前的 State 供反向用；反向先算 `dtau`，再把 `dstate` 乘 `sech2`。
- **无 mask 算子**：`output_final_state=False` 或 `mask=None` 时调用独立
  no-mask kernel（chunk 边界无条件 SN）。`mask=None` 且 `output_final_state=True`
  时 Python 入口发双语 UserWarning 并返回 `(out, None)`。
- **推理/任意长度**：`generalized_delta_rule_sn_inference` 按 chunk 读 tau，
  `T` 不要求被 16 整除；任意长度 prefill 也可用单步
  `generalized_delta_rule_sn_single_step`（每步算 SN，按 per-sample `do_sn` 选择）。
- **分片**：sharding rule 已按输出实际维度重建 `NamedSharding`；mask 无 head 维，
  在 head 轴自动 replicate；Torch 侧按 head 独立 launch，切好 head 维即可 TP。
- **测试 tau 生成**：`x ~ N(7.0, 0.5)`（见 tests/conftest.py），
  `tau = softplus(x) + 1.0`，使 tau ≈ 1000，近似恒等映射。

### RWKV-6（`rwkv6_kernel`）

- `ops_rwkv_kernel.py` 的 `RWKVKernelOperator` 是数值 ground truth（while_loop
  逐步 RNN，State 全程 fp32）；`native_keras_op.py` 是函数式薄封装。
- `MAX_SEQUENCE_LENGTH` 是**编译期常量**：JAX 构建目录 `build_{head}_{len}/`，
  改参数须删旧目录重编；运行时 `T > MAX_SEQUENCE_LENGTH` 显式 raise。
- `initial_state` 可为 `[H,N,N]` 或 `[B,H,N,N]`；batch 维为 1 或 B 时自动推断
  `state_map`，否则须显式传 `[B]` int。
- `KERNEL_TYPE="triton"` 未实现，**静默回退 native**。
- Torch 版输入在 CPU 时回退 native；要求张量 contiguous 且同 device/dtype。

### mHC（`mhc_kernel`）

- **`C` 必须被 128 整除**（代码无 assert，不满足直接 kernel 报错）。
- 只支持 `native` / `triton`；triton 只替换 `mhc_pre_op_fused` 与 `mhc_post_op`
  两个底层符号，高层封装始终共用 native 的 `linear_and_reshape`。
- **显存优化靠手写 VJP 重计算**：pre_op 反向在 kernel 内重跑 sinkhorn，前向只
  保存未归一化原始输入。**改 sinkhorn 数学必须同步改 fwd+bwd 两个 kernel。**
- jax_triton 反向 grid 固定 `(total_bt, 1)`（消除原子操作的关键，勿改二维）。
- mhc pre 输出 `H_post/H_res` 强制 float32；`M = phi.shape[-1]` 须为 32 的倍数。

## 6. 构建与清理

- 构建后端 hatchling；wheel 为纯 Python（`py3-none-any`），CUDA/Triton 全部
  **运行时 JIT 编译**，发布物不含二进制 kernel。构建：`python -m build`。
- **版本号两处同步**：`pyproject.toml` 与 `rwkv_ops/__init__.py` 的 `__version__`。
- `MANIFEST.in` 包含 `*.cu/*.cpp/*.h/CMakeLists.txt` 等源码；新增编译产物类型要
  同步加排除规则；**不要手改 `dist/`**。
- `clean_build_artifacts.clean_all()` 删除各 `jax_cuda_kernel*/build_*` 与
  `*.so`、ninja 日志、所有 `__pycache__`；`tests/conftest.py` 在 pytest 会话结束
  自动调用——**每次跑完 pytest，JAX CUDA 测试下次会重新编译**（耗时是预期的）。
  新增带 FFI 的算子目录时把它的 `build_*`/`*.so` 加进 `_CLEAN_PATTERNS`。

## 7. 测试

- **分目录跑**：同一 pytest 会话只能锁一个 Keras 后端。直接
  `pytest tests/jax/...` 即可（子目录 conftest 用 setdefault 设后端）；
  `KERNEL_TYPE` 不进环境变量，全部经 fixture 显式传参。在仓库根目录运行
  （测试经根 conftest 的 `sys.path` 注入 import `tests.conftest`）。
- markers：`torch` / `jax` / `numpy` / `tensorflow` / `slow`（编译类慢测试）。
- **根 `tests/conftest.py` 刻意不 import torch/jax/keras/rwkv_ops**，避免收集
  阶段锁定后端；后端 import 下沉到子目录 conftest / 测试函数。
- fixtures：`rwkv7_inputs`（r/k/v ~ N(0,1)；a、b 为同一 z 的 ±单位向量；
  `w = -softplus(w_raw) - 0.5`）；`rwkv7_sn_inputs` 加 tau（见 §5）。
- 断言：`assert_allclose_with_stats`（打印 exact/close/max/mean diff，判定只看
  atol/rtol）。容差惯例：
  - RWKV-7 前向 y atol=1e-5/rtol=1e-2（SN 的 y 放宽到 1e-4）；
    final_state atol=1e-5/rtol=1e-3；反向 grad atol=7e-3/rtol=1e-3
    （grad_b 放宽到 1e-2）。
  - RWKV-6 / mHC：一律 1e-2/1e-2。
- 加速后端测试统一模式：同一组 numpy 输入分别喂 native 参考和加速算子，输入
  cast bf16（state/tau/mask 保持 f32），loss 用 `mean(y²)+mean(state²)`，
  `head_first` 参数化 [False, True]。
- SN 类算子必须镜像覆盖：带/不带 mask 的前向+反向（含 tau 梯度）、无 mask 警告
  + None state、任意长度推理、不规则 padding、head 轴 TP（jax）。
- **测试文件名唯一性**：`tests/` 各子目录无 `__init__.py`，跨目录同名会
  `import file mismatch`；统一带后端前缀（`test_jax_*` 等）。

## 8. 常见陷阱

1. RWKV-6 `cuda` 后端用 `jax.ffi`，需 JAX ≥ 0.4.31。
2. RWKV-7 Triton 后端主要验证 HEAD_SIZE=64；其他 head_size 用 `cuda`。
3. chunkwise RWKV-7 要求 T 被 16 整除，否则静默出错或 UB。
4. RWKV-6 的 `MAX_SEQUENCE_LENGTH` 是编译期常量，改后必须删旧 build 目录。
5. mHC 的 `C` 必须被 128 整除。
6. 不要手改 `dist/` 或 `build/` 内容。
7. `custom_partitioning` 在 eager 下也 trace 内层：涉及运行期探测（如 pallas
   autotune）的逻辑必须在 trace 之外完成（见 §4 Pallas warmup 约定）。
8. JAX CUDA 测试每次会话后 `.so` 被自动清理，下次重编译是预期行为。
9. 递推类 kernel 的数值对拍必须用仓库 fixture 的稳定输入分布；随手造的随机
   数据会让 delta-rule 递推指数发散，f32 参考自身都会差出 1e5，无法用于判定。

## 9. 提交前检查清单

- [ ] `ruff check .` 与 `ruff format --check .` 全绿。
- [ ] 版本号两处同步（`pyproject.toml` + `rwkv_ops/__init__.py`）。
- [ ] 新增编译产物已加入 `.gitignore` / `MANIFEST.in` 排除/包含规则。
- [ ] 新算子/新后端已在 `rwkv_ops/__init__.py` 正确暴露。
- [ ] 对应后端测试已补充（文件名带后端前缀），并验证与 native 数值一致。
- [ ] Triton kernel 改动已同步检查 jax/torch 两个桥接。
- [ ] 新 FFI 算子的构建目录已加入 `clean_build_artifacts._CLEAN_PATTERNS`。
- [ ] 已更新 `AGENTS.md`、`README.md`、`ENREADME.md` 的支持矩阵。

## 10. 新增算子指南

1. `rwkv_ops/` 下新建目录，至少含 `__init__.py`（工厂分发）和 `native_*.py`。
2. 每个加速后端写桥接文件，共享同一套 Triton/CUDA kernel 源。
3. 在 `rwkv_ops/__init__.py` 导入并实例化、暴露。
4. `tests/<backend>/` 加测试（文件名带后端前缀），fixture 挂进对应 conftest。
5. 更新 `AGENTS.md` 与两个 README 的支持矩阵。

## 11. 文件速查

| 文件 | 作用 |
|---|---|
| `rwkv_ops/__init__.py` | 包入口、环境变量解析、API 暴露（import 即实例化全部算子） |
| `rwkv_ops/pallas_utils.py` | Pallas 后端公共机制：候选配置、autotune、SPMD 辅助 |
| `rwkv_ops/rwkv7_kernel/__init__.py` | RWKV-7 后端分发器 |
| `rwkv_ops/rwkv7_kernel/native_keras_op.py` | RWKV-7 原生参考实现 |
| `rwkv_ops/rwkv7_kernel/triton_kernel.py` | RWKV-7 共享 Triton kernel |
| `rwkv_ops/rwkv7_sn_kernel/` | SN 版，结构与 rwkv7_kernel 平行 |
| `rwkv_ops/rwkv6_kernel/ops_rwkv_kernel.py` | RWKV-6 数值 ground truth |
| `rwkv_ops/mhc_kernel/native_op.py` | mHC 原生参考实现 |
| `rwkv_ops/cuda_tools/nvcc_wrap` | nvcc 包装器（绕过 CUDA 13.1/glibc 冲突），勿删 |
| `clean_build_artifacts.py` | 编译产物清理（pytest 会话结束自动调用） |
| `pyproject.toml` | 包元数据（hatchling）、依赖、版本号 |
