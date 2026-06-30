"""
RWKV-6 JAX FFI CUDA kernel（函数式接口）。

编译/加载逻辑与 rwkv7_kernel/jax_cuda_kernel/wkv7_jax.py 对齐，
核心 CUDA 计算复用 rwkv6_kernel/torch_cuda_kernel/wkv6_cuda.cu 的等价实现。
"""

from __future__ import annotations

import ctypes
import pathlib
import subprocess
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.custom_partitioning import custom_partitioning

_CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

# ---------------------------------------------------------------------------
# Shardy 分片规则（与 rwkv7 风格一致）
# 字母含义：b=Batch, t=Time, c=Channel(H*N), h=Head, n=HeadDim
# ---------------------------------------------------------------------------
FWD_RULE = "b t c, b t c, b t c, b t c, h n -> b t c"
BWD_RULE = "b t c, b t c, b t c, b t c, h n, b t c -> b t c, b t c, b t c, b t c, h n"
FWD_STATE_RULE = (
    "b t c, b t c, b t c, b t c, h n, b, b h n n -> b t c, b h n n"
)


def _fwd_infer_sharding(arg_shapes, arg_shardings):
    return (arg_shardings[0],)


def _bwd_infer_sharding(arg_shapes, arg_shardings):
    return (
        arg_shardings[0],
        arg_shardings[1],
        arg_shardings[2],
        arg_shardings[3],
        arg_shardings[4],
    )


def _fwd_state_infer_sharding(arg_shapes, arg_shardings):
    # y 跟随 r，final_state 跟随 init_state
    return (arg_shardings[0], arg_shardings[6])


def _create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)

        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition


def get_jax_rwkv6(head_size: int = 64, max_sequence_length: int = 4096):
    _BUILD_DIR = _CURRENT_DIR / f"build_{head_size}_{max_sequence_length}"
    _SO_PATH = _CURRENT_DIR / f"build_{head_size}_{max_sequence_length}" / "wkv6.so"

    def _ensure_compiled() -> pathlib.Path:
        if _SO_PATH.exists():
            return _SO_PATH

        print("[rwkv6_jax] First use – compiling CUDA kernel…")
        src_dir = _CURRENT_DIR
        build_dir = _BUILD_DIR
        build_dir.mkdir(parents=True, exist_ok=True)

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
            f"-DHEAD_SIZE={head_size}",
            f"-DMAX_SEQUENCE_LENGTH={max_sequence_length}",
            f"-DCMAKE_CUDA_FLAGS={' '.join(cuda_flags)}",
        ]
        subprocess.check_call(cmake_args)
        subprocess.check_call(["cmake", "--build", str(build_dir), "-j"])
        subprocess.check_call(["cmake", "--install", str(build_dir)])

        if not _SO_PATH.exists():
            raise RuntimeError("Compilation failed – wkv6.so not found.")

        print("[rwkv6_jax] Compilation finished – output at", _SO_PATH)
        return _SO_PATH

    _lib = ctypes.CDLL(_ensure_compiled())

    jax.ffi.register_ffi_target(
        "wkv6_fwd", jax.ffi.pycapsule(_lib.Wkv6Fwd), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "wkv6_bwd", jax.ffi.pycapsule(_lib.Wkv6Bwd), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "wkv6_fwd_with_state",
        jax.ffi.pycapsule(_lib.Wkv6FwdWithState),
        platform="CUDA",
    )

    def _transpose_head(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
        if head_first:
            return jnp.transpose(x, (0, 2, 1, 3))
        return x

    def _transpose_head_back(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
        if head_first:
            return jnp.transpose(x, (0, 2, 1, 3))
        return x

    # -----------------------------------------------------------------------
    # 前向 kernel（无状态，可反传）
    # -----------------------------------------------------------------------
    def _rwkv6_fwd_impl(r, k, v, w, u):
        B, T, C = r.shape
        dtype = r.dtype
        out_type = jax.ShapeDtypeStruct((B, T, C), dtype)
        return jax.ffi.ffi_call(
            "wkv6_fwd", out_type, vmap_method="broadcast_all"
        )(r, k, v, w, u)

    @custom_partitioning
    def _rwkv6_fwd(r, k, v, w, u):
        return _rwkv6_fwd_impl(r, k, v, w, u)

    _rwkv6_fwd.def_partition(
        infer_sharding_from_operands=_fwd_infer_sharding,
        sharding_rule=FWD_RULE,
        partition=_create_partition(_rwkv6_fwd_impl),
    )

    @jax.custom_vjp
    def _rwkv6(r, k, v, w, u):
        return _rwkv6_fwd(r, k, v, w, u)

    def _fwd(r, k, v, w, u):
        y = _rwkv6_fwd(r, k, v, w, u)
        return y, (r, k, v, w, u)

    def _rwkv6_bwd_impl(r, k, v, w, u, gy):
        B, T, C = r.shape
        dtype = r.dtype
        gr_type = jax.ShapeDtypeStruct((B, T, C), dtype)
        gk_type = jax.ShapeDtypeStruct((B, T, C), dtype)
        gv_type = jax.ShapeDtypeStruct((B, T, C), dtype)
        gw_type = jax.ShapeDtypeStruct((B, T, C), dtype)
        gu_type = jax.ShapeDtypeStruct((B, C), dtype)
        return jax.ffi.ffi_call(
            "wkv6_bwd", (gr_type, gk_type, gv_type, gw_type, gu_type), vmap_method="broadcast_all"
        )(r, k, v, w, u, gy)

    @custom_partitioning
    def _rwkv6_bwd(r, k, v, w, u, gy):
        return _rwkv6_bwd_impl(r, k, v, w, u, gy)

    _rwkv6_bwd.def_partition(
        infer_sharding_from_operands=_bwd_infer_sharding,
        sharding_rule=BWD_RULE,
        partition=_create_partition(_rwkv6_bwd_impl),
    )

    def _bwd(res, gy):
        r, k, v, w, u = res
        gy = jnp.asarray(gy, jnp.bfloat16)
        gr, gk, gv, gw, gu = _rwkv6_bwd(r, k, v, w, u, gy)
        # CUDA kernel 返回的 gu 形状为 (B, C)，按 batch 求和后 reshape 回 (H, N)
        H, N = u.shape
        gu = jnp.sum(gu, axis=0).reshape((H, N))
        return gr, gk, gv, gw, gu

    _rwkv6.defvjp(_fwd, _bwd)

    # -----------------------------------------------------------------------
    # 前向 kernel（带初始状态/最终状态，仅前向）
    # -----------------------------------------------------------------------
    def _rwkv6_fwd_with_state_impl(r, k, v, w, u, state_map, init_state):
        B, T, C = r.shape
        dtype = r.dtype
        out_type = jax.ShapeDtypeStruct((B, T, C), dtype)
        state_type = jax.ShapeDtypeStruct((B, C // head_size, head_size, head_size), dtype)
        return jax.ffi.ffi_call(
            "wkv6_fwd_with_state", (out_type, state_type), vmap_method="broadcast_all"
        )(r, k, v, w, u, state_map, init_state)

    @custom_partitioning
    def _rwkv6_fwd_with_state(r, k, v, w, u, state_map, init_state):
        return _rwkv6_fwd_with_state_impl(r, k, v, w, u, state_map, init_state)

    _rwkv6_fwd_with_state.def_partition(
        infer_sharding_from_operands=_fwd_state_infer_sharding,
        sharding_rule=FWD_STATE_RULE,
        partition=_create_partition(_rwkv6_fwd_with_state_impl),
    )

    # -----------------------------------------------------------------------
    # 公共 API
    # -----------------------------------------------------------------------
    def rwkv6_op(
        r: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        w: jnp.ndarray,
        u: jnp.ndarray,
        initial_state: Optional[jnp.ndarray] = None,
        output_final_state: bool = False,
        state_map: Optional[jnp.ndarray] = None,
        head_first: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        dtype = r.dtype

        r = _transpose_head(r, head_first)
        k = _transpose_head(k, head_first)
        v = _transpose_head(v, head_first)
        w = _transpose_head(w, head_first)

        if dtype != jnp.bfloat16:
            from ..native_keras_op import rwkv6 as native_rwkv6

            y = native_rwkv6(
                r=r,
                k=k,
                v=v,
                w=w,
                u=u,
                initial_state=initial_state,
                output_final_state=output_final_state,
                state_map=state_map,
                head_size=head_size,
                max_sequence_length=max_sequence_length,
            )
            if output_final_state:
                y, final_state = y
                y = _transpose_head_back(y, head_first)
                return y, final_state
            y = _transpose_head_back(y, head_first)
            return y

        r = jnp.asarray(r, dtype=jnp.bfloat16)
        k = jnp.asarray(k, dtype=jnp.bfloat16)
        v = jnp.asarray(v, dtype=jnp.bfloat16)
        w = jnp.asarray(w, dtype=jnp.bfloat16)
        u = jnp.asarray(u, dtype=jnp.bfloat16)

        B, T, C = r.shape
        if T > max_sequence_length:
            raise ValueError(
                f"序列长度 T={T} 超过编译期最大序列长度 max_sequence_length={max_sequence_length}"
            )
        if C % head_size != 0:
            raise ValueError(
                f"通道数 C={C} 必须能被 head_size={head_size} 整除"
            )
        H = C // head_size
        N = head_size

        # u 统一 reshape 为 (H, N)
        u = jnp.reshape(u, (H, N))

        has_initial_state = initial_state is not None
        if has_initial_state:
            if len(initial_state.shape) == 3:
                initial_state = initial_state[None, :]
            if initial_state.shape[1:] != (H, N, N):
                raise ValueError(
                    f"initial_state 形状必须为 (B, H, N, N) 或 (H, N, N)，"
                    f"当前为 {initial_state.shape}"
                )
            initial_state = jnp.asarray(initial_state, dtype=jnp.bfloat16)

            if state_map is None:
                state_kinds = initial_state.shape[0]
                if state_kinds == 1:
                    state_map = jnp.zeros((B,), dtype=jnp.int32)
                elif state_kinds == B:
                    state_map = jnp.arange(B, dtype=jnp.int32)
                else:
                    raise ValueError(
                        "无法推断 state_map，请手动指定"
                    )
            else:
                state_map = jnp.asarray(state_map, dtype=jnp.int32)
                if state_map.shape != (B,):
                    raise ValueError(f"state_map 形状必须为 (B,)，当前为 {state_map.shape}")
        else:
            assert state_map is None, "指定 state_map 时必须同时传入 initial_state"
            initial_state = jnp.zeros((1, H, N, N), dtype=jnp.bfloat16)
            state_map = jnp.zeros((B,), dtype=jnp.int32)

        if not has_initial_state and not output_final_state:
            y = _rwkv6(r, k, v, w, u)
        else:
            y, final_state = _rwkv6_fwd_with_state(
                r, k, v, w, u, state_map, initial_state
            )

        y = _transpose_head_back(y, head_first)
        y = jnp.asarray(y, dtype)

        if output_final_state:
            return y, final_state
        return y

    return rwkv6_op
