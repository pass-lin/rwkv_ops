import os
from functools import lru_cache
from typing import Literal
import functools
import triton
import jax
import jax.numpy as jnp
from enum import Enum
import contextlib


@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)[
        "multiprocessor_count"
    ]


@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        import warnings

        warnings.warn(
            ("Triton is not supported on current platform, roll back to CPU."),
            stacklevel=1,
        )
        return "cpu"


@lru_cache(maxsize=None)
def _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:
    device = get_available_device()
    if device == "cuda":
        return "nvidia"
    elif device == "hip":
        return "amd"
    elif device == "xpu":
        return "intel"
    else:
        return device


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
device = get_available_device() if get_available_device() != "hip" else "cuda"

device_platform = _check_platform()

is_intel = device_platform == "intel"
is_nvidia = device_platform == "nvidia"
is_amd = device_platform == "amd"

use_cuda_graph = is_nvidia and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"


device = get_available_device() if get_available_device() != "hip" else "cuda"

is_intel_a770 = False
device = jax.devices()
is_tf32_supported = is_nvidia


def get_all_max_shared_memory():
    return [
        triton.runtime.driver.active.utils.get_device_properties(i)["max_shared_mem"]
        for i in range(len(jax.devices()))
    ]


device_shared_mem_list = get_all_max_shared_memory()


@lru_cache(maxsize=None)
def is_triton_shared_mem_enough(
    max_shared_mem: int = 102400, tensor_idx: int = 0
) -> bool:
    max_shared_memory = device_shared_mem_list[tensor_idx]
    return max_shared_memory >= max_shared_mem


device_capacity = is_triton_shared_mem_enough()


def _cpu_device_warning():
    import warnings

    warnings.warn(
        ("Triton is not supported on current platform, roll back to CPU."), stacklevel=1
    )


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)[
                "max_shared_mem"
            ]
            for i in range(len(jax.devices()))
        ]
    except BaseException:
        _cpu_device_warning()
        return [-1]


class Backend(Enum):
    ADA = 101376  # RTX 4090
    AMPERE = 166912  # A100
    HOPPER = 232448  # H100
    DEFAULT = 102400  # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@lru_cache(maxsize=None)
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


def tensor_cache(fn):
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., jax.Array]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., jax.Array]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args = None
    last_kwargs = None
    last_result = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens):
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens, chunk_size: int):
    indices = jnp.concatenate(
        [
            jnp.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )

    return jnp.stack([jnp.cumsum(jnp.equal(indices, 0), 0) - 1, indices], 1)


def input_guard(fn):
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i for i in args)
        contiguous_kwargs = {k: v for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, jax.Array):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, jax.Array):
                    tensor = value
                    break

        if tensor is not None:
            ctx = tensor.device
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


is_intel_alchemist = False
