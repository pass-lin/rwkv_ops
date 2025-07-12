from ..triton_kernel.cumsum import *
import jax_triton as jt
import jax
import triton


def chunk_rwkv6_fwd_cumsum(
    g: jax.Array,
    chunk_size: int,
) -> jax.Array:
    B, T, H, S = g.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)

    out_shapes = [
        jax.ShapeDtypeStruct(g.shape, "float32"),
        jax.ShapeDtypeStruct(g.shape, "float32"),
    ]

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    gi, ge = jt.triton_call(
        g,
        T,
        H=H,
        S=S,
        BT=BT,
        grid=grid,
        kernel=chunk_rwkv6_fwd_cumsum_kernel,
        out_shape=out_shapes,
    )

    return gi, ge
