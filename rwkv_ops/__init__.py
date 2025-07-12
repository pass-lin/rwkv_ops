__version__ = "0.1.1"
import os

KERNEL_TYPE = os.environ.get("KERNEL_TYPE", "triton")
KERAS_BACKEND = os.environ.get("KERAS_BACKEND")
BACKEND = os.environ.get("KERNEL_BACKEND")


if KERAS_BACKEND is not None:
    BACKEND = KERAS_BACKEND
elif BACKEND is not None:
    os.environ["KERAS_BACKEND"] = BACKEND
else:
    import torch
    import keras

    BACKEND = "torch"
    os.environ["KERAS_BACKEND"] = BACKEND
    keras.config.set_backend("torch")
assert KERNEL_TYPE in ["triton", "cuda", "native"]
assert BACKEND in ["torch", "jax", "numpy", "tensorflow"]
from .rwkv7_kernel import get_generalized_delta_rule

generalized_delta_rule, RWKV7_USE_KERNEL = get_generalized_delta_rule(
    KERNEL_TYPE=KERNEL_TYPE
)
rwkv7_op = generalized_delta_rule
