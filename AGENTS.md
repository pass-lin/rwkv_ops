# Agent Notes for rwkv_ops

## 代码风格

- 运行 `ruff check .` 与 `ruff format --check .`，确保无格式/排版错误。
- 禁止 `from ... import *`；使用显式 import。
- 禁止给 lambda 赋值（`grid = lambda ...`），改用 `def`。
- 函数式风格：每个 kernel 的 Python 入口都返回 `out` 或 `(out, final_state)`。

## RWKV-7 State Norm (`rwkv7_sn_kernel`)

### 设计约定

- **tau 语义**：CUDA kernel 只接收预处理后的 `tau`，即 `tau = softplus(param) + 1.0`，必须严格大于 1。
- **padding chunk**：全 padding 的 chunk 必须将对应 `tau` 置 0.0；kernel 内部通过 `tau > 0` 判断是否执行 SN。
- **无内置 token-level mask**：调用者需在外部保证 padding 位置 `k=0, a=0, w=-inf`。
- **输出与 State 的关系**：输出始终基于 SN **之前** 的 State；SN 只修改传递给下一步/下一 chunk 的 State。
- **训练版本**：只在 chunk 边界（每 16 tokens）执行 SN；native 用 `ops.cond`，CUDA 在 `(t+1)%16==0` 处分支。
- **单步版本**：每步都计算 SN，再用 `ops.where` / CUDA 分支按 per-sample `do_sn` 选择。

### CUDA 实现要点

- 输入 `r,w,k,v,a,b` 强制 cast 到 `bfloat16`，State 和 `tau` 保持 `float32`。
- 训练 kernel 的 checkpoint `s_` 保存 **SN 之前** 的 State，供反向使用。
- 反向时先算 `dtau`，再把 `dstate`/`dstateT` 乘 `sech2`。
- Torch 通过 `torch.utils.cpp_extension.load` 延迟编译；JAX 通过 `cmake` + `cuda_tools/nvcc_wrap` 编译 FFI `.so`。

### 训练与推理 kernel 的取舍

- 训练 kernel 返回的 `s` / `sa` 是为反向传播保留的 checkpoint；即使外层不做梯度，
  在 FFI/CUDA 层这些仍是显式输出并占用显存。
- 推理 / prefill kernel（`generalized_delta_rule_sn_inference`）只输出 `y` 与最终 state，
  可显著降低显存，但当前实现仍按 chunk 读取 `tau`，因此 **T 仍需被 16 整除**。
- 若需要任意长度 prefill，请使用单步 RNN 接口 `generalized_delta_rule_sn_single_step`。

### 测试

- `tests/torch/test_torch_rwkv7_sn.py` 与 `tests/jax/test_jax_rwkv7_sn.py` 必须覆盖：
  - 前向输出与 final_state 的 CUDA vs native 对比。
  - 反向梯度（含 `tau`）的 CUDA vs native 有限差分/自动微分对比。
  - 单步 RNN 与 native 单步在 16 步内一致。
- tau 生成：`x ~ N(4.0, 0.5)`，然后 `tau = softplus(x) + 1.0`，使 tau 接近 100，近似恒等映射。

## 构建与清理

- JAX FFI 构建产物位于 `rwkv_ops/*/jax_cuda_kernel*/build_*` 和对应 `.so`，由 `clean_build_artifacts.clean_all()` 在测试后自动清理。
- CUDA 13.1 与新版 glibc 的 `rsqrt`/`rsqrtf` 冲突通过 `cuda_tools/nvcc_wrap` 绕过，不要修改系统 CUDA 头文件。
