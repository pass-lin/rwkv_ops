"""
RWKV-6 数值校对脚本：CUDA kernel vs native Keras ops ground truth。

覆盖：
- 前向输出 y 与最终状态 final_state
- 反向梯度 (gr, gk, gv, gw, gu)
- 带 initial_state / state_map 的状态传播

支持后端：torch / jax；支持的 dtype 视 CUDA kernel 而定。
"""

import argparse
import os
import sys
import numpy as np


def set_seed(seed=42):
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _to_numpy(x):
    """统一把后端张量转成 numpy。"""
    if hasattr(x, "detach"):
        # PyTorch（先在 CPU；bf16 不支持直接 numpy，转 float32）
        import torch

        x = x.detach().cpu()
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.numpy()
    if hasattr(x, "numpy"):
        # JAX / TensorFlow
        return np.array(x)
    return np.array(x)


def _make_inputs(backend, B, T, H, N, dtype, device=None):
    C = H * N
    rng = np.random.default_rng(42)
    r = rng.standard_normal((B, T, C), dtype=np.float32)
    k = rng.standard_normal((B, T, C), dtype=np.float32)
    v = rng.standard_normal((B, T, C), dtype=np.float32)
    w = rng.standard_normal((B, T, C), dtype=np.float32) * 0.5
    u = rng.standard_normal((H, N), dtype=np.float32) * 0.1
    init_state = rng.standard_normal((B, H, N, N), dtype=np.float32) * 0.1

    if backend == "torch":
        import torch

        torch_dtype = getattr(torch, dtype)
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        t = lambda a: torch.tensor(
            a, dtype=torch_dtype, device=dev, requires_grad=False
        )
        return t(r), t(k), t(v), t(w), t(u), t(init_state)

    if backend == "jax":
        import jax.numpy as jnp

        jnp_dtype = getattr(jnp, dtype)
        return tuple(
            jnp.asarray(a, dtype=jnp_dtype) for a in (r, k, v, w, u, init_state)
        )

    raise ValueError(f"不支持的后端: {backend}")


def _metrics(ref, tgt):
    ref = _to_numpy(ref).astype(np.float64)
    tgt = _to_numpy(tgt).astype(np.float64)
    diff = np.abs(ref - tgt)
    return {
        "max": float(np.max(diff)),
        "mean": float(np.mean(diff)),
        "rel_max": float(np.max(diff / (np.abs(ref) + 1e-8))),
        "rel_mean": float(np.mean(diff / (np.abs(ref) + 1e-8))),
    }


def _print_metrics(label, ref, tgt):
    m = _metrics(ref, tgt)
    print(
        f"  {label}: max={m['max']:.3e}, mean={m['mean']:.3e}, "
        f"rel_max={m['rel_max']:.3e}, rel_mean={m['rel_mean']:.3e}"
    )
    return m


def _run_torch(B, T, H, N, dtype):
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  Torch 未检测到 GPU，CUDA 分支会回退到 native；跳过 CUDA 校对。")
        return

    from rwkv_ops import get_rwkv6_kernel

    native_op = get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="native")
    cuda_op = get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="cuda", MAX_SEQUENCE_LENGTH=T)

    # native 以 float32 作为数值 ground truth；CUDA 使用目标 dtype
    r_ref, k_ref, v_ref, w_ref, u_ref, init_ref = _make_inputs(
        "torch", B, T, H, N, "float32", device
    )
    r_c, k_c, v_c, w_c, u_c, init_c = _make_inputs("torch", B, T, H, N, dtype, device)

    print(f"\n[Torch | cuda_dtype={dtype}] forward + state (ref=float32)")
    y_n, s_n = native_op(
        r_ref,
        k_ref,
        v_ref,
        w_ref,
        u_ref,
        initial_state=init_ref,
        output_final_state=True,
    )
    y_c, s_c = cuda_op(
        r_c,
        k_c,
        v_c,
        w_c,
        u_c,
        initial_state=init_c,
        output_final_state=True,
    )
    _print_metrics("y", y_n, y_c)
    _print_metrics("final_state", s_n, s_c)

    print(f"[Torch | cuda_dtype={dtype}] state_map (ref=float32)")
    init_map_ref = init_ref[:1]
    init_map_c = init_c[:1]
    state_map = [0] * B
    y_n2, s_n2 = native_op(
        r_ref,
        k_ref,
        v_ref,
        w_ref,
        u_ref,
        initial_state=init_map_ref,
        output_final_state=True,
        state_map=state_map,
    )
    y_c2, s_c2 = cuda_op(
        r_c,
        k_c,
        v_c,
        w_c,
        u_c,
        initial_state=init_map_c,
        output_final_state=True,
        state_map=state_map,
    )
    _print_metrics("y", y_n2, y_c2)
    _print_metrics("final_state", s_n2, s_c2)

    print(f"[Torch | cuda_dtype={dtype}] backward (ref=float32)")

    def _grads(op, r, k, v, w, u):
        r_t = r.clone().requires_grad_(True)
        k_t = k.clone().requires_grad_(True)
        v_t = v.clone().requires_grad_(True)
        w_t = w.clone().requires_grad_(True)
        u_t = u.clone().requires_grad_(True)
        y = op(r_t, k_t, v_t, w_t, u_t)
        loss = (y.float() ** 2).sum()
        loss.backward()
        return r_t.grad, k_t.grad, v_t.grad, w_t.grad, u_t.grad

    gr_n, gk_n, gv_n, gw_n, gu_n = _grads(native_op, r_ref, k_ref, v_ref, w_ref, u_ref)
    gr_c, gk_c, gv_c, gw_c, gu_c = _grads(cuda_op, r_c, k_c, v_c, w_c, u_c)
    for label, gn, gc in zip(
        ["gr", "gk", "gv", "gw", "gu"],
        [gr_n, gk_n, gv_n, gw_n, gu_n],
        [gr_c, gk_c, gv_c, gw_c, gu_c],
    ):
        _print_metrics(label, gn, gc)


def _run_jax(B, T, H, N, dtype):
    import jax
    import jax.numpy as jnp

    from rwkv_ops import get_rwkv6_kernel

    native_op = get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="native")
    cuda_op = get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="cuda", MAX_SEQUENCE_LENGTH=T)

    # native 以 float32 作为数值 ground truth；CUDA 使用目标 dtype
    r_ref, k_ref, v_ref, w_ref, u_ref, init_ref = _make_inputs(
        "jax", B, T, H, N, "float32"
    )
    r_c, k_c, v_c, w_c, u_c, init_c = _make_inputs("jax", B, T, H, N, dtype)
    u_for_cuda = jnp.reshape(u_c, (H, N))

    print(f"\n[JAX | cuda_dtype={dtype}] forward + state (ref=float32)")
    y_n, s_n = native_op(
        r_ref,
        k_ref,
        v_ref,
        w_ref,
        u_ref,
        initial_state=init_ref,
        output_final_state=True,
    )
    y_c, s_c = cuda_op(
        r_c,
        k_c,
        v_c,
        w_c,
        u_for_cuda,
        initial_state=init_c,
        output_final_state=True,
    )
    _print_metrics("y", y_n, y_c)
    _print_metrics("final_state", s_n, s_c)

    print(f"[JAX | cuda_dtype={dtype}] state_map (ref=float32)")
    init_map_ref = init_ref[:1]
    init_map_c = init_c[:1]
    state_map = jnp.zeros((B,), dtype=jnp.int32)
    y_n2, s_n2 = native_op(
        r_ref,
        k_ref,
        v_ref,
        w_ref,
        u_ref,
        initial_state=init_map_ref,
        output_final_state=True,
        state_map=state_map,
    )
    y_c2, s_c2 = cuda_op(
        r_c,
        k_c,
        v_c,
        w_c,
        u_for_cuda,
        initial_state=init_map_c,
        output_final_state=True,
        state_map=state_map,
    )
    _print_metrics("y", y_n2, y_c2)
    _print_metrics("final_state", s_n2, s_c2)

    print(f"[JAX | cuda_dtype={dtype}] backward (ref=float32)")

    def _loss(op, r, k, v, w, u):
        y = op(r, k, v, w, u)
        return jnp.sum(jnp.asarray(y, jnp.float32) ** 2)

    # native JAX 实现使用 dynamic while_loop，无法直接 reverse-mode 求导；
    # 这里用随机方向的有限差分作为参考方向导数，与 CUDA custom_vjp 的方向导数对比。
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    dirs = [
        jax.random.normal(k, p.shape, dtype=jnp.float32)
        for k, p in zip(keys, (r_ref, k_ref, v_ref, w_ref, u_ref))
    ]

    def _directional_fd(op, params, dirs, eps=1e-4):
        def loss_of(params):
            return _loss(op, *params)

        plus = [p + eps * d for p, d in zip(params, dirs)]
        minus = [p - eps * d for p, d in zip(params, dirs)]
        return (loss_of(plus) - loss_of(minus)) / (2 * eps)

    ref_directional = _directional_fd(
        native_op, [r_ref, k_ref, v_ref, w_ref, u_ref], dirs
    )

    loss_cuda = lambda r, k, v, w, u: _loss(cuda_op, r, k, v, w, u)
    grad_c = jax.grad(loss_cuda, argnums=(0, 1, 2, 3, 4))(
        r_c, k_c, v_c, w_c, u_for_cuda
    )
    cuda_directional = sum(
        jnp.sum(jnp.asarray(g, jnp.float32) * d) for g, d in zip(grad_c, dirs)
    )
    diff = float(jnp.abs(ref_directional - cuda_directional))
    rel = float(diff / (jnp.abs(ref_directional) + 1e-8))
    print(
        f"  directional derivative: ref={float(ref_directional):.6e}, "
        f"cuda={float(cuda_directional):.6e}, abs_diff={diff:.3e}, rel_diff={rel:.3e}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["torch", "jax"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Torch 支持 float32/bfloat16/float16；JAX CUDA 当前仅支持 bfloat16",
    )
    args = parser.parse_args()

    B = args.batch
    T = args.seq
    H = args.heads
    N = args.head_size

    os.environ["KERAS_BACKEND"] = args.backend
    os.environ["KERNEL_TYPE"] = "native"

    if args.dtype is None:
        args.dtype = "bfloat16" if args.backend == "jax" else "float32"

    if args.backend == "torch":
        _run_torch(B, T, H, N, args.dtype)
    else:
        _run_jax(B, T, H, N, args.dtype)

    print("\n✅ RWKV-6 CUDA vs native 校对完成")


if __name__ == "__main__":
    main()
