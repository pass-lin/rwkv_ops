import triton
import triton.language as tl

is_gather_supported = hasattr(triton.language, "gather")
if not is_gather_supported:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        # This is a fallback implementation when tl.gather is not supported
        # In order to pass triton compiler, there is no actual gather operation
        return src
else:
    gather = tl.gather
exp = tl.exp
import keras

if keras.backend.backend() == "jax":
    from ..get_jax_devices_info import *
else:
    from ..get_torch_devices_info import *
