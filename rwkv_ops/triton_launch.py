"""Triton kernel 的固定 launch 参数与 JAX-Triton 调用封装。"""

from __future__ import annotations

import os

# recurrent kernel 沿时间维串行展开，value 方向切得太宽会同时放大单个 program 的
# 串行工作量与寄存器压力，因此默认只切一小段 value 轴。
MAX_BV = int(os.environ.get("RWKV_OPS_TRITON_MAX_BV", "32"))
NUM_WARPS = 4
NUM_STAGES = 3


def _unwrap_autotune(kernel):
    """拆出 Autotuner 背后的裸 kernel 与其首个 config 的 constexpr 取值。

    Args:
        kernel: Triton ``Autotuner`` 或裸 kernel。

    Returns:
        tuple: ``(bare_kernel, config_kwargs)``；裸 kernel 的 ``config_kwargs`` 为空。
    """
    if not (hasattr(kernel, "configs") and hasattr(kernel, "fn")):
        return kernel, {}
    config = kernel.configs[0]
    return kernel.fn, dict(config.kwargs)


def triton_call(*args, kernel, out_shape, grid, **kwargs):
    """以固定 launch 参数调用 Triton kernel。

    ``jax_triton`` 遇到 ``triton.autotune`` 会在真实张量上把每个 config 各编译、
    计时一遍；对时间维串行的 recurrent kernel 来说，一次调用会被放大成十余次完整
    编译。这里取 Autotuner 背后的 JITFunction 并固定 num_warps/num_stages，
    torch 侧的 autotune 保持不变。Autotune config 里的 constexpr（如 RWKV-7 的
    ``MINI_BSZ``）必须补回调用参数，否则 kernel 与 grid 都拿不到该值。

    Args:
        *args: kernel 输入，顺序必须与 Triton 签名逐参数对齐。
        kernel: Triton kernel；传 ``Autotuner`` 时改用其背后的 JITFunction。
        out_shape: ``jax.ShapeDtypeStruct`` 序列，顺序与输出指针对应。
        grid: launch grid，常量元组或 ``callable(meta) -> tuple``。
        **kwargs: 透传给 ``jax_triton.triton_call`` 的其余参数。

    Returns:
        按 ``out_shape`` 声明的输出；只有一个输出时为数组本身，否则为元组。

    Raises:
        ImportError: 未安装 ``jax-triton``。
    """
    import jax_triton as jt

    kernel, config_kwargs = _unwrap_autotune(kernel)
    for name, value in config_kwargs.items():
        kwargs.setdefault(name, value)
    kwargs.setdefault("num_warps", NUM_WARPS)
    kwargs.setdefault("num_stages", NUM_STAGES)
    return jt.triton_call(
        *args, **kwargs, kernel=kernel, out_shape=out_shape, grid=grid
    )
