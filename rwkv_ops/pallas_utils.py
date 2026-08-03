"""
Pallas 后端公共工具：后端配置、autotune、SPMD 辅助。

设计原则：
- kernel 本体只使用稳定的公开 Pallas API（pl.pallas_call / pl.BlockSpec /
  pl.program_id / ref 索引 / lax.fori_loop / jnp），不写死任何后端私有 API，
  以兼容老版本 jax 的 Triton 后端、未来版本（Mosaic GPU）以及 TPU。
- 后端与编译参数通过 PallasConfig 选择，并带一个简单的 autotune：
  首次遇到新 shape key 时在 eager 模式下对候选配置计时，缓存最快者；
  无法 lowering 的候选自动跳过。可用 RWKV_OPS_PALLAS_AUTOTUNE=0 关闭，
  关闭后按候选顺序取第一个可编译的配置。
- custom_partitioning 即使在 eager 调用下也会 trace 内层函数，因此配置
  解析必须发生在 eager 层（ensure_config），被 trace 的路径只查缓存。
"""

from __future__ import annotations

import os
import time
from typing import NamedTuple

import jax
import jax.tree_util as jtu
from jax.experimental import pallas as pl

PALLAS_AUTOTUNE = os.environ.get("RWKV_OPS_PALLAS_AUTOTUNE", "1") == "1"


class PallasConfig(NamedTuple):
    backend: str  # "triton" | "mgpu"
    num_warps: int = 4
    num_stages: int = 2


def candidate_configs() -> list[PallasConfig]:
    """候选配置，按优先级排序；autotune 从中选最快，否则取首个可编译者。

    triton 后端排在前面：jax 0.10.x 的 Mosaic GPU lowering 对逐行动态索引
    有 128 元素向量约束，多数版本下无法编译本仓库的 kernel；此时 autotune
    会自动跳过 mgpu 候选。在未来修复了该限制的 jax 版本上，mgpu 可凭计时
    结果胜出。
    """
    configs = []
    try:
        from jax.experimental.pallas import triton as _  # noqa: F401

        for num_warps in (4, 8):
            for num_stages in (2, 3):
                configs.append(PallasConfig("triton", num_warps, num_stages))
    except Exception:
        pass
    try:
        from jax.experimental.pallas import mosaic_gpu as _  # noqa: F401

        configs.append(PallasConfig("mgpu"))
    except Exception:
        pass
    return configs


def whole_specs(n: int, backend: str) -> list[pl.BlockSpec]:
    """整数组 BlockSpec。

    - mgpu：必须显式放 GMEM（默认会物化到 SMEM，整序列 block 放不下）。
    - triton / TPU：使用默认 memory_space，保持最大兼容性。
    """
    if backend == "mgpu":
        from jax.experimental.pallas import mosaic_gpu as plgpu

        return [pl.BlockSpec(memory_space=plgpu.MemorySpace.GMEM) for _ in range(n)]
    return [pl.BlockSpec(memory_space=pl.ANY) for _ in range(n)]


def make_pallas_call(kernel, n_in, out_shape, config: PallasConfig, grid):
    kwargs = {
        "grid": grid,
        "in_specs": whole_specs(n_in, config.backend),
        "out_specs": whole_specs(len(out_shape), config.backend),
        "out_shape": out_shape,
    }
    if config.backend == "triton":
        from jax.experimental.pallas import triton as pltriton

        kwargs["compiler_params"] = pltriton.CompilerParams(
            num_warps=config.num_warps, num_stages=config.num_stages
        )
    return pl.pallas_call(kernel, **kwargs)


# (kernel_name, shapes/dtypes) -> PallasConfig
_config_cache: dict = {}


def _config_key(name, args, out_shape):
    return (
        name,
        tuple((a.shape, a.dtype) for a in args),
        tuple((o.shape, o.dtype) for o in out_shape),
    )


def _probe_config(config, kernel, n_in, out_shape, grid, args, reps=3):
    """编译并计时一个候选配置；失败抛异常由调用方过滤。"""
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
    configs = candidate_configs()
    best_cfg, best_time = None, float("inf")
    for cfg in configs:
        try:
            dt = _probe_config(
                cfg,
                kernel,
                n_in,
                out_shape,
                grid,
                args,
                reps=3 if PALLAS_AUTOTUNE else 1,
            )
        except Exception:
            continue
        if not PALLAS_AUTOTUNE:
            best_cfg = cfg
            break
        if dt < best_time:
            best_cfg, best_time = cfg, dt
    if best_cfg is None:
        raise RuntimeError(
            "rwkv_ops pallas: 没有任何候选后端可以编译该 kernel "
            f"(尝试了 {[c.backend for c in configs]})"
        )
    return best_cfg


def ensure_config(name, kernel, out_shape, grid, args):
    """eager 模式下解析并缓存后端配置；traced 时跳过（由 launch 走默认）。"""
    key = _config_key(name, args, out_shape)
    if key in _config_cache:
        return
    if any(isinstance(a, jax.core.Tracer) for a in jtu.tree_leaves(args)):
        return
    _config_cache[key] = _resolve_config(key, kernel, len(args), out_shape, grid, args)


def launch(name, kernel, out_shape, grid, args):
    """选择后端配置并执行 pallas_call。args 为 head-first 的输入数组元组。"""
    key = _config_key(name, args, out_shape)
    if key not in _config_cache:
        # jit/custom_partitioning 追踪期间无法探测，用默认候选
        _config_cache[key] = candidate_configs()[0]
    config = _config_cache[key]
    call = make_pallas_call(kernel, len(args), out_shape, config, grid)
    return call(*args)


def create_partition(impl_fn):
    """custom_partitioning 的 partition 回调（逐设备本地执行 impl_fn）。"""

    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)

        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition
