__version__ = "0.5.5"
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
assert BACKEND in ["torch", "jax", "numpy", "tensorflow"]
from .rwkv7_kernel import get_generalized_delta_rule, get_rnn_generalized_delta_rule
from .rwkv6_kernel import get_rwkv6_kernel


generalized_delta_rule, generalized_delta_rule_inference, RWKV7_USE_TRITON_KERNEL = (
    get_generalized_delta_rule(KERNEL_TYPE=KERNEL_TYPE)
)
rwkv7_op = generalized_delta_rule
rwkv7_op_inference = generalized_delta_rule_inference

rnn_generalized_delta_rule = get_rnn_generalized_delta_rule(KERNEL_TYPE=KERNEL_TYPE)
rwkv7_op_rnn = rnn_generalized_delta_rule


RWKV6_OP = get_rwkv6_kernel(KERNEL_TYPE=KERNEL_TYPE)
