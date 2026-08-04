"""
Pallas 后端公共工具：后端探测、参数 autotune、SPMD 辅助。

设计原则：
- kernel 本体只使用稳定的公开 Pallas API（pl.pallas_call / pl.BlockSpec /
  pl.program_id / ref 索引 / lax.fori_loop / jnp），不写死任何后端私有 API，
  以兼容老版本 jax（GPU 上只有 Triton lowering）、新版本 jax（默认 Mosaic GPU
  lowering）以及 TPU。
- **后端选择是确定性的能力探测，不做计时择优**：按偏好顺序（默认后端 →
  triton 后端）逐个编译探测，第一个能 lowering 的即为本机后端。jax < 0.9 的
  GPU pallas 只有 triton lowering，探测自然落到 triton；未来 triton 后端被
  移除后探测自然落到默认后端。可用 RWKV_OPS_PALLAS_BACKEND=default|mgpu|triton
  强制覆盖（调试用）。
- **autotune 只调已选定后端的性能参数**（对齐 @triton.autotune 的语义，按
  shape key 缓存最优）：triton 后端调 num_warps × num_stages；MGPU 调
  lowering_semantics / reduction_scratch_bytes。RWKV_OPS_PALLAS_AUTOTUNE=0
  关闭后用默认参数。
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
_PALLAS_BACKEND_ENV = os.environ.get("RWKV_OPS_PALLAS_BACKEND", "").lower()


class PallasConfig(NamedTuple):
    """一次 pallas_call 的完整配置：后端 + 后端相关参数（可哈希，作缓存 key）。"""

    backend: str  # "default" | "mgpu" | "triton"
    params: tuple = ()  # 例如 (("num_warps", 4), ("num_stages", 2))


def _triton_available() -> bool:
    try:
        from jax.experimental.pallas import triton as _  # noqa: F401

        return True
    except Exception:
        return False


def _mgpu_available() -> bool:
    try:
        from jax.experimental.pallas import mosaic_gpu as _  # noqa: F401

        return True
    except Exception:
        return False


def _backend_preference() -> list[str]:
    """后端探测顺序。env 覆盖优先；否则 默认后端 -> triton。"""
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
    """某个后端下参与 autotune 的参数组合（首个为默认参数）。"""
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
    # "default"：不强行注入参数（老版本 jax 的默认 lowering 不接受 compiler_params）
    return [()]


def whole_specs(n: int, backend: str) -> list[pl.BlockSpec]:
    """整数组 BlockSpec。

    - MGPU lowering（mgpu 后端，或新版本 jax 的 default）：必须显式放 GMEM
      （默认会物化到 SMEM，整序列 block 放不下）。
    - triton 后端 / 老版本 jax / TPU：使用默认 memory_space，保持最大兼容性。
    """
    if backend in ("mgpu", "default") and _mgpu_available():
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
    return (
        name,
        tuple((a.shape, a.dtype) for a in args),
        tuple((o.shape, o.dtype) for o in out_shape),
    )


def _probe_config(config, kernel, n_in, out_shape, grid, args, reps=3):
    """编译并计时一个配置；失败抛异常由调用方过滤。"""
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
    """两段式解析：先按偏好顺序探测后端（默认参数），再在该后端内 autotune。"""
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
        # jit/custom_partitioning 追踪期间无法探测，用偏好顺序首个后端 + 默认参数
        _config_cache[key] = PallasConfig(_backend_preference()[0])
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
