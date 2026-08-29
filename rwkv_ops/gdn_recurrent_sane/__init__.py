"""Gated DeltaNet recurrent SANE 算子后端分发器。"""

import keras
from ..utils import _use_triton


def _use_jax_triton(KERNEL_TYPE):
    """jax + GPU 平台且 KERNEL_TYPE=triton 时启用 Triton kernel。"""
    if not _use_triton(KERNEL_TYPE):
        return False
    try:
        import jax
        import jax_triton  # noqa: F401
        import triton  # noqa: F401
    except Exception:
        return False
    return jax.devices()[0].platform == "gpu"


def get_gated_delta_net_recurrent_sane(KERNEL_TYPE="native"):
    """按后端与 KERNEL_TYPE 返回 Gated DeltaNet recurrent SANE 训练算子。

    Args:
        KERNEL_TYPE: str，"native" / "pallas" / "cuda" / "triton"。
            torch 后端 "native"/"triton" 走 Triton；jax 后端 "native"/"pallas" 在
            GPU/TPU 上走 Pallas；缺硬件静默回退 native。

    Returns:
        Callable，签名与 `native_keras_op.gated_delta_net_recurrent_sane` 一致。
    """
    from .native_keras_op import gated_delta_net_recurrent_sane

    if keras.config.backend() == "torch" and KERNEL_TYPE in ("native", "triton"):
        if _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import (
                gated_delta_net_recurrent_sane as triton_op,
            )

            return triton_op
    elif keras.config.backend() == "jax":
        from ..pallas_utils import _use_jax_pallas

        if KERNEL_TYPE == "triton" and _use_jax_triton(KERNEL_TYPE):
            from .jax_triton_kernel import (
                gated_delta_net_recurrent_sane as triton_op,
            )

            return triton_op
        if _use_jax_pallas(KERNEL_TYPE):
            from .jax_pallas_kernel import gated_delta_net_recurrent_sane as pallas_op

            return pallas_op

    return gated_delta_net_recurrent_sane


def get_gated_delta_net_recurrent_sane_inference(KERNEL_TYPE="native"):
    """按后端与 KERNEL_TYPE 返回 Gated DeltaNet recurrent SANE 推理算子。

    Args:
        KERNEL_TYPE: str，"native" / "pallas" / "cuda" / "triton"。
            torch 后端 "native"/"triton" 走 Triton；jax 后端 "native"/"pallas" 在
            GPU/TPU 上走 Pallas；缺硬件静默回退 native。

    Returns:
        Callable，签名与 `native_keras_op.gated_delta_net_recurrent_sane_inference` 一致。
    """
    from .native_keras_op import gated_delta_net_recurrent_sane_inference

    if keras.config.backend() == "torch" and KERNEL_TYPE in ("native", "triton"):
        if _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import (
                gated_delta_net_recurrent_sane_inference as triton_op,
            )

            return triton_op
    elif keras.config.backend() == "jax":
        from ..pallas_utils import _use_jax_pallas

        if KERNEL_TYPE == "triton" and _use_jax_triton(KERNEL_TYPE):
            from .jax_triton_kernel import (
                gated_delta_net_recurrent_sane_inference as triton_op,
            )

            return triton_op
        if _use_jax_pallas(KERNEL_TYPE):
            from .jax_pallas_kernel import (
                gated_delta_net_recurrent_sane_inference as pallas_op,
            )

            return pallas_op

    return gated_delta_net_recurrent_sane_inference


def get_gated_delta_net_recurrent_sane_single_step(KERNEL_TYPE="native"):
    """按后端与 KERNEL_TYPE 返回 Gated DeltaNet recurrent SANE 单步 RNN 算子。

    Args:
        KERNEL_TYPE: str，"native" / "pallas" / "cuda" / "triton"。
            torch 后端 "native"/"triton" 走 Triton；jax 后端 "native"/"pallas" 在
            GPU/TPU 上走 Pallas；缺硬件静默回退 native。

    Returns:
        Callable，签名与 `native_keras_op.gated_delta_net_recurrent_sane_single_step` 一致。
    """
    from .native_keras_op import gated_delta_net_recurrent_sane_single_step

    if keras.config.backend() == "torch" and KERNEL_TYPE in ("native", "triton"):
        if _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import (
                gated_delta_net_recurrent_sane_single_step as triton_op,
            )

            return triton_op
    elif keras.config.backend() == "jax":
        from ..pallas_utils import _use_jax_pallas

        if KERNEL_TYPE == "triton" and _use_jax_triton(KERNEL_TYPE):
            from .jax_triton_kernel import (
                gated_delta_net_recurrent_sane_single_step as triton_op,
            )

            return triton_op
        if _use_jax_pallas(KERNEL_TYPE):
            from .jax_pallas_kernel import (
                gated_delta_net_recurrent_sane_single_step as pallas_op,
            )

            return pallas_op

    return gated_delta_net_recurrent_sane_single_step
