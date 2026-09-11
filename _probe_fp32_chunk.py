"""临时探针：实测 delta_net_chunk triton fp32 输入行为与误差。"""

import numpy as np
import torch

from rwkv_ops.delta_net_chunk.native_keras_op import delta_net_chunk as native_chunk
from rwkv_ops.delta_net_chunk.torch_triton_kernel import (
    delta_net_chunk as triton_chunk,
)

rng = np.random.default_rng(42)
B, T, H, K, V = 2, 128, 4, 64, 128
q = rng.standard_normal((B, T, H, K), dtype=np.float32)
k = rng.standard_normal((B, T, H, K), dtype=np.float32)
v = rng.standard_normal((B, T, H, V), dtype=np.float32)
beta = (1.0 / (1.0 + np.exp(-rng.standard_normal((B, T, H), dtype=np.float32)))).astype(
    np.float32
)
h0 = rng.standard_normal((B, H, K, V), dtype=np.float32) * 0.1

dev = "cuda:0"


def to_t(x, dtype=torch.float32, grad=False):
    return torch.from_numpy(x).to(dev).to(dtype).requires_grad_(grad)


def stats(name, ref, tgt):
    d = (ref.detach().float() - tgt.detach().float()).abs()
    rel = d / ref.detach().float().abs().clamp_min(1e-6)
    print(f"{name}: max_diff={d.max().item():.3e} mean_diff={d.mean().item():.3e} "
          f"max_rel={rel.max().item():.3e}")


for cs in (16, 32, 64):
    out_n, st_n = native_chunk(
        to_t(q), to_t(k), to_t(v), to_t(beta),
        initial_state=to_t(h0), output_final_state=True, chunk_size=cs,
    )
    out_t, st_t = triton_chunk(
        to_t(q), to_t(k), to_t(v), to_t(beta),
        initial_state=to_t(h0), output_final_state=True, chunk_size=cs,
    )
    print(f"chunk_size={cs} out dtype: native={out_n.dtype} triton={out_t.dtype}")
    stats(f"cs={cs} out", out_n, out_t)
    stats(f"cs={cs} state", st_n, st_t)


def run_and_grad(op, cs):
    q_t, k_t, v_t = to_t(q, grad=True), to_t(k, grad=True), to_t(v, grad=True)
    beta_t, h0_t = to_t(beta, grad=True), to_t(h0, grad=True)
    out, state = op(
        q_t, k_t, v_t, beta_t,
        initial_state=h0_t, output_final_state=True, chunk_size=cs,
    )
    (out.pow(2).mean() + state.pow(2).mean()).backward()
    return [t.grad for t in (q_t, k_t, v_t, beta_t, h0_t)]


for cs in (16, 32):
    gn = run_and_grad(native_chunk, cs)
    gt = run_and_grad(triton_chunk, cs)
    for name, ref, tgt in zip(["q", "k", "v", "beta", "h0"], gn, gt):
        stats(f"cs={cs} bwd {name}", ref, tgt)
