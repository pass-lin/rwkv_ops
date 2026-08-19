"""Pallas 后端公共工具：后端探测、参数 autotune、SPMD 辅助。"""

from __future__ import annotations

import os
import time
from typing import NamedTuple

import jax
import jax.tree_util as jtu
from jax.experimental import pallas as pl
from .utils import _force_keras_native

PALLAS_AUTOTUNE = os.environ.get("RWKV_OPS_PALLAS_AUTOTUNE", "1") == "1"
_PALLAS_BACKEND_ENV = os.environ.get("RWKV_OPS_PALLAS_BACKEND", "").lower()


class PallasConfig(NamedTuple):
    """一次 pallas_call 的完整配置。

    Args:
        backend: str, "default" | "mgpu" | "triton"。
        params: tuple, 后端相关参数键值对（可哈希，作缓存 key）。
    """

    backend: str  # "default" | "mgpu" | "triton"
    params: tuple = ()  # 例如 (("num_warps", 4), ("num_stages", 2))


def _triton_available() -> bool:
    """检查 Triton Pallas 后端是否可导入。"""
    try:
        from jax.experimental.pallas import triton as _  # noqa: F401

        return True
    except Exception:
        return False


def _mgpu_available() -> bool:
    """检查 Mosaic GPU Pallas 后端是否可导入。"""
    try:
        from jax.experimental.pallas import mosaic_gpu as _  # noqa: F401

        return True
    except Exception:
        return False


def _backend_preference() -> list[str]:
    """返回后端探测顺序，环境变量覆盖优先。"""
    if _PALLAS_BACKEND_ENV in ("default", "mgpu", "triton"):
        rest = [b for b in ("default", "mgpu", "triton") if b != _PALLAS_BACKEND_ENV]
        return [_PALLAS_BACKEND_ENV] + rest
    order = ["default"]
    if _triton_available():
        order.append("triton")
    if _mgpu_available():
        order.append("mgpu")
    return order


def _param_grid(backend: str) -> list[tuple]:
    """返回指定后端下参与 autotune 的参数组合（首个为默认参数）。"""
    if backend == "triton":
        return [(("num_warps", w), ("num_stages", s)) for w in (4, 8) for s in (2, 3)]
    if backend == "mgpu":
        grid = [()]
        if _mgpu_available():
            from jax.experimental.pallas import mosaic_gpu as plgpu

            if hasattr(plgpu, "LoweringSemantics"):
                for sem in ("Lane", "Warpgroup"):
                    if hasattr(plgpu.LoweringSemantics, sem):
                        grid.append((("lowering_semantics", sem),))
            grid.append((("reduction_scratch_bytes", 16384),))
        return grid
    # "default" 后端不注入 compiler_params，保持与默认 lowering 的兼容。
    return [()]


def whole_specs(n: int, backend: str) -> list[pl.BlockSpec]:
    """构造 n 个整数组输入/输出的 BlockSpec。

    MGPU lowering 下必须显式放到 GMEM，避免默认 SMEM 放不下整序列 block；
    triton 后端/TPU 使用默认 memory_space 保持兼容。

    Args:
        n: int, BlockSpec 数量。
        backend: str, 后端名称。

    Returns:
        list[pl.BlockSpec], BlockSpec 列表。
    """
    if backend in ("mgpu", "default") and _mgpu_available():
        from jax.experimental.pallas import mosaic_gpu as plgpu

        return [pl.BlockSpec(memory_space=plgpu.MemorySpace.GMEM) for _ in range(n)]
    return [pl.BlockSpec(memory_space=pl.ANY) for _ in range(n)]


def make_pallas_call(kernel, n_in, out_shape, config: PallasConfig, grid):
    """按 PallasConfig 组装 pl.pallas_call 的参数并调用。

    Args:
        kernel: callable, Pallas kernel 函数。
        n_in: int, 输入数量。
        out_shape: tuple, 输出形状/dtype 元组。
        config: PallasConfig, 后端配置。
        grid: tuple, pallas_call grid。

    Returns:
        callable, 已包装的 pallas_call。
    """
    kwargs = {
        "grid": grid,
        "in_specs": whole_specs(n_in, config.backend),
        "out_specs": whole_specs(len(out_shape), config.backend),
        "out_shape": out_shape,
    }
    params = dict(config.params)
    if config.backend == "triton":
        from jax.experimental.pallas import triton as pltriton

        kwargs["compiler_params"] = pltriton.CompilerParams(**params)
    elif config.backend == "mgpu" and params:
        from jax.experimental.pallas import mosaic_gpu as plgpu

        if "lowering_semantics" in params:
            params["lowering_semantics"] = getattr(
                plgpu.LoweringSemantics, params["lowering_semantics"]
            )
        kwargs["compiler_params"] = plgpu.CompilerParams(**params)
    return pl.pallas_call(kernel, **kwargs)


# (kernel_name, shapes/dtypes) -> PallasConfig
_config_cache: dict = {}


def _config_key(name, args, out_shape):
    """生成 (kernel 名称, 输入 shape/dtype, 输出 shape/dtype) 的缓存 key。"""
    return (
        name,
        tuple((a.shape, a.dtype) for a in args),
        tuple((o.shape, o.dtype) for o in out_shape),
    )


def _probe_config(config, kernel, n_in, out_shape, grid, args, reps=3):
    """编译并计时一个 PallasConfig；失败由调用方捕获过滤。"""
    call = make_pallas_call(kernel, n_in, out_shape, config, grid)
    fn = jax.jit(call)
    out = fn(*args)
    jax.block_until_ready(out)
    reps = max(reps, 1)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps


def _resolve_config(key, kernel, n_in, out_shape, grid, args):
    """按偏好顺序探测可用后端，并在选定后端内 autotune 参数。"""
    tried = []
    for backend in _backend_preference():
        default = PallasConfig(backend)
        try:
            _probe_config(default, kernel, n_in, out_shape, grid, args, reps=1)
        except Exception:
            tried.append(backend)
            continue
        # 后端可用：在其参数网格内选最快（autotune 关闭时直接用默认参数）
        if not PALLAS_AUTOTUNE:
            return default
        best_cfg, best_time = default, float("inf")
        for params in _param_grid(backend):
            cfg = PallasConfig(backend, params)
            try:
                dt = _probe_config(cfg, kernel, n_in, out_shape, grid, args)
            except Exception:
                continue
            if dt < best_time:
                best_cfg, best_time = cfg, dt
        return best_cfg
    raise RuntimeError(
        "rwkv_ops pallas: 没有任何候选后端可以编译该 kernel "
        f"(尝试了 {tried or _backend_preference()})"
    )


def ensure_config(name, kernel, out_shape, grid, args):
    """eager 模式下解析并缓存后端配置；traced 时跳过探测。

    真实数组调用必须在 trace 之前完成，因为 custom_partitioning 在 eager 下
    也会 trace 内层函数，被 trace 的路径只查缓存。

    Args:
        name: str, kernel 名称（缓存 key 的一部分）。
        kernel: callable, Pallas kernel 函数。
        out_shape: tuple, 输出形状/dtype 元组。
        grid: tuple, pallas_call grid。
        args: tuple, 真实输入数组元组。
    """
    key = _config_key(name, args, out_shape)
    if key in _config_cache:
        return
    if any(isinstance(a, jax.core.Tracer) for a in jtu.tree_leaves(args)):
        return
    _config_cache[key] = _resolve_config(key, kernel, len(args), out_shape, grid, args)


def launch(name, kernel, out_shape, grid, args):
    """选择后端配置并执行 pallas_call。

    Args:
        name: str, kernel 名称（缓存 key 的一部分）。
        kernel: callable, Pallas kernel 函数。
        out_shape: tuple, 输出形状/dtype 元组。
        grid: tuple, pallas_call grid。
        args: tuple, head-first 的输入数组元组。

    Returns:
        tuple, pallas_call 输出。
    """
    key = _config_key(name, args, out_shape)
    if key not in _config_cache:
        # jit/custom_partitioning 追踪期间无法探测，用偏好顺序首个后端 + 默认参数
        _config_cache[key] = PallasConfig(_backend_preference()[0])
    config = _config_cache[key]
    call = make_pallas_call(kernel, len(args), out_shape, config, grid)
    return call(*args)


def _use_jax_pallas(KERNEL_TYPE):
    """jax + GPU/TPU 且 KERNEL_TYPE=native 时启用 Pallas kernel。"""
    if KERNEL_TYPE != "native" or _force_keras_native():
        return False
    try:
        import jax
    except Exception:
        return False
    return jax.devices()[0].platform in ("gpu", "tpu")


def create_partition(impl_fn):
    """生成 custom_partitioning 的 partition 回调。

    回调在逐设备本地执行 impl_fn，并传播输入/输出的 sharding。

    Args:
        impl_fn: callable, 实现函数。

    Returns:
        callable, partition 回调函数。
    """

    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)

        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition
