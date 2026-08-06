"""RWKV-OPS 包入口：解析环境变量并暴露全部算子 API。"""

__version__ = "0.9.0"
import os

KERNEL_TYPE = os.environ.get("KERNEL_TYPE", "cuda").lower()
KERAS_BACKEND = os.environ.get("KERAS_BACKEND")
BACKEND = os.environ.get("KERNEL_BACKEND")


if KERAS_BACKEND is not None:
    BACKEND = KERAS_BACKEND.lower()
elif BACKEND is not None:
    os.environ["KERAS_BACKEND"] = BACKEND.lower()
else:
    import keras

    BACKEND = "torch"
    os.environ["KERAS_BACKEND"] = BACKEND
    keras.config.set_backend("torch")
assert KERNEL_TYPE in ["triton", "cuda", "native"]
assert BACKEND in ["torch", "jax", "numpy", "tensorflow", "openvino"]
from .rwkv7_kernel import (  # noqa: E402
    get_generalized_delta_rule,
    get_rnn_generalized_delta_rule,
)
from .rwkv7_sane_kernel import (  # noqa: E402
    get_generalized_delta_rule_sane,
    get_rnn_generalized_delta_rule_sane,
)

from .rwkv6_kernel import get_rwkv6_kernel  # noqa: E402
from .mhc_kernel import get_mhc_kernel  # noqa: E402

generalized_delta_rule, generalized_delta_rule_inference = get_generalized_delta_rule(
    KERNEL_TYPE=KERNEL_TYPE
)
rwkv7_op = generalized_delta_rule
rwkv7_op_inference = generalized_delta_rule_inference

rnn_generalized_delta_rule = get_rnn_generalized_delta_rule(KERNEL_TYPE=KERNEL_TYPE)
rwkv7_op_rnn = rnn_generalized_delta_rule

(
    generalized_delta_rule_sane,
    generalized_delta_rule_sane_inference,
) = get_generalized_delta_rule_sane(KERNEL_TYPE=KERNEL_TYPE)
rwkv7_op_sane = generalized_delta_rule_sane
rwkv7_op_sane_inference = generalized_delta_rule_sane_inference

rnn_generalized_delta_rule_sane = get_rnn_generalized_delta_rule_sane(
    KERNEL_TYPE=KERNEL_TYPE
)
rwkv7_op_sane_rnn = rnn_generalized_delta_rule_sane


rwkv6_op = get_rwkv6_kernel(KERNEL_TYPE=KERNEL_TYPE)
# 保留旧名称的向后兼容别名
RWKV6_OP = rwkv6_op
mhc_pre_op, mhc_post_op = get_mhc_kernel(KERNEL_TYPE=KERNEL_TYPE)
