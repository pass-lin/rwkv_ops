"""JAX 版 RWKV7-SANE 单步 wkv kernel（仅前向）。"""

from __future__ import annotations
import pathlib
import subprocess
import ctypes
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Optional, Tuple, Union

from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

_CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
_NVCC_WRAPPER = _CURRENT_DIR.parents[1] / "cuda_tools" / "nvcc_wrap"

_REGISTERED_FFI_TARGET: str | None = None

#  SPMD 切分规则（Einsum 风格）
# b=Batch, h=Head, k/m=HeadDim。支持 DP（batch 维）与 TP（head 维）。
# tau 为 [B, H]（per-head），do_sane 为 [B]（per-sample）。
FWD_RULE = "b h k, b h k, b h k, b h k, b h k, b h k, b h, b, b h k m -> b h k, b h k m"


def _q_spec(qs):
    spec = getattr(qs, "spec", None)
    if spec is None or len(spec) != 3:
        return None
    return spec


def _sharding_like_q(qs):
    """为 y (B, H, K) 构造与输入一致的 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(*spec))


def _sharding_for_state(qs):
    """为最终 State (B, H, K, K) 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], spec[2], spec[2]))


def _fwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[1]  # q
    return (_sharding_like_q(qs), _sharding_for_state(qs))


def _create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)

        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition


def get_jax_generalized_delta_rule_sane_single_step(HEAD_SIZE=64, chunk_size: int = 16):
    # chunk_size 仅用于签名一致，单步 kernel 内部固定 T=1，忽略该值。
    """返回 RWKV-7-SANE 单步（T=1）JAX-CUDA FFI 算子。

    Args:
        HEAD_SIZE: int，head 维度大小，必须为 4 的倍数。

    Returns:
        single_step_op：函数，输入 T=1，输出 (y, final_state)。
    """
    _BUILD_DIR = _CURRENT_DIR / f"build_single_step_{HEAD_SIZE}"
    _SO_PATH = _BUILD_DIR / "wkv7_sane_single_step.so"

    def _ensure_compiled() -> pathlib.Path:
        if _SO_PATH.exists():
            return _SO_PATH

        print("[rwkv7_sane_single_step_jax] First use – compiling CUDA kernel…")
        src_dir = _CURRENT_DIR
        build_dir = _BUILD_DIR
        build_dir.mkdir(exist_ok=True)

        xla_include_dir = jax.ffi.include_dir()
        if not xla_include_dir:
            raise RuntimeError("jax.ffi.include_dir() 返回空，请检查 JAX >= 0.4.31")

        cuda_flags = [
            "-ftz=true",
            "-prec-div=false",
            "-prec-sqrt=false",
            "--use_fast_math",
            "-O3",
            "-Xptxas=-O3",
            "-res-usage",
            "--extra-device-vectorization",
            f"-D_C_={HEAD_SIZE}",
        ]

        cmake_args = [
            "cmake",
            "-S",
            str(src_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={_CURRENT_DIR}",
            f"-DXLA_INCLUDE_DIR={xla_include_dir}",
            f"-DCMAKE_CUDA_COMPILER={_NVCC_WRAPPER}",
            f"-DCMAKE_CUDA_FLAGS={' '.join(cuda_flags)}",
        ]
        subprocess.check_call(cmake_args)
        subprocess.check_call(["cmake", "--build", str(build_dir), "-j"])
        subprocess.check_call(["cmake", "--install", str(build_dir)])

        if not _SO_PATH.exists():
            raise RuntimeError(
                "Compilation failed – wkv7_sane_single_step.so not found."
            )

        print("[rwkv7_sane_single_step_jax] Compilation finished – output at", _SO_PATH)
        return _SO_PATH

    global _REGISTERED_FFI_TARGET
    _lib = ctypes.CDLL(_ensure_compiled())
    _target_name = f"wkv7_sane_single_step_fwd_{HEAD_SIZE}"
    if _REGISTERED_FFI_TARGET != _target_name:
        jax.ffi.register_ffi_target(
            _target_name,
            jax.ffi.pycapsule(_lib.Wkv7SaneSingleStepFwd),
            platform="CUDA",
        )
        _REGISTERED_FFI_TARGET = _target_name

    def _transpose_head(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
        if head_first:
            return jnp.transpose(x, (0, 2, 1, 3))
        x = jnp.asarray(x, dtype=jnp.bfloat16)
        return x

    def _wkv7_sane_single_step_impl(w, q, k, v, a, b, tau, do_sane, h0):
        B, H, K = q.shape
        dtype = q.dtype
        out_type = jax.ShapeDtypeStruct((B, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, K, K), jnp.float32)

        y, s = jax.ffi.ffi_call(
            _target_name,
            (out_type, s_type),
            vmap_method="broadcast_all",
        )(w, q, k, v, a, b, tau, do_sane, h0)
        return y, s

    @custom_partitioning
    def _wkv7_sane_single_step_spmd(w, q, k, v, a, b, tau, do_sane, h0):
        return _wkv7_sane_single_step_impl(w, q, k, v, a, b, tau, do_sane, h0)

    _wkv7_sane_single_step_spmd.def_partition(
        infer_sharding_from_operands=_fwd_infer_sharding,
        sharding_rule=FWD_RULE,
        partition=_create_partition(_wkv7_sane_single_step_impl),
    )

    def generalized_delta_rule_sane_single_step(
        r: jnp.ndarray,
        w: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        tau: jnp.ndarray,
        do_sane,
        initial_state: Optional[jnp.ndarray] = None,
        output_final_state: bool = True,
        head_first: bool = False,
        chunk_size: int = 16,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """RWKV-7-SANE 单步推理（JAX-CUDA FFI 入口）。

        Args:
            r, w, k, v, a, b: [B, 1, H, K]（head_first=False）或 [B, H, 1, K]（head_first=True）。
            tau: [B, H]，float32，必须 > 0。
            do_sane: [B]，bool 或 int32。非 0 表示对该 sample 执行 SANE。
            initial_state: [B, H, K, K] 或 [1, H, K, K]，float32，可选。
            output_final_state: bool，是否返回最终 state。
            head_first: bool，输入输出是否 head 维优先。

        Returns:
            out: [B, 1, H, K]（或 [B, H, 1, K]），bfloat16。
            final_state: [B, H, K, K]，float32。output_final_state=False 时不返回。

        Raises:
            ValueError: 时间维不为 1。
        """
        r = _transpose_head(r, head_first)
        w = _transpose_head(w, head_first)
        k = _transpose_head(k, head_first)
        v = _transpose_head(v, head_first)
        a = _transpose_head(a, head_first)
        b = _transpose_head(b, head_first)
        tau = jnp.asarray(tau, jnp.float32)

        B, T, H, K = r.shape
        if T != 1:
            raise ValueError(f"Single-step kernel requires T=1, but got T={T}.")

        if initial_state is None:
            h0 = jnp.zeros((B, H, K, K), jnp.float32)
        else:
            h0 = jnp.asarray(initial_state, jnp.float32)

        r = r[:, 0, :, :]
        w = w[:, 0, :, :]
        k = k[:, 0, :, :]
        v = v[:, 0, :, :]
        a = a[:, 0, :, :]
        b = b[:, 0, :, :]

        # JAX 默认关闭 x64，CUDA 侧也使用 int32（仅 0/1）。
        if isinstance(do_sane, bool):
            do_sane = jnp.full((B,), int(do_sane), dtype=jnp.int32)
        else:
            do_sane = jnp.asarray(do_sane, dtype=jnp.int32)

        out, last_state = _wkv7_sane_single_step_spmd(
            w, r, k, v, a, b, tau, do_sane, h0
        )
        out = jnp.expand_dims(out, axis=1)
        out = jnp.asarray(out, r.dtype)

        if output_final_state:
            return out, last_state
        return out

    return generalized_delta_rule_sane_single_step
