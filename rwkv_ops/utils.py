import keras
import os


def _force_keras_native():
    """RWKV_OPS_KERAS_NATIVE=1 时强制 native 使用纯 Keras ops。"""
    v = os.environ.get("RWKV_OPS_KERAS_NATIVE", "").lower()
    return v not in ("", "0", "false")


def _use_triton(KERNEL_TYPE):
    """torch + 非 CPU 平台且 KERNEL_TYPE=native 时启用 Triton kernel。"""
    if keras.config.backend() == "torch":
        import torch

        if KERNEL_TYPE != "native" or _force_keras_native():
            return False
        try:
            import torch
            import triton  # noqa: F401
        except Exception:
            return False
        if torch.cuda.is_available():
            return True
        xpu = getattr(torch, "xpu", None)
        if xpu is not None:
            try:
                return bool(xpu.is_available())
            except Exception:
                return False
        return False
    elif keras.config.backend() == "jax":
        if KERNEL_TYPE == "triton":
            import jax

            return jax.devices()[0].platform == "gpu"
    return False
