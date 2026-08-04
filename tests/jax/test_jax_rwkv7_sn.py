"""RWKV-7 State Neutralization JAX CUDA kernel 数值测试。"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tests.conftest import assert_allclose_with_stats


def _to_jax(arr, dtype):
    """把 numpy 数组转成指定 dtype 的 JAX 数组。

    Args:
        arr: np.ndarray，输入数组。
        dtype: str, jnp dtype 名称。

    Returns:
        jax.Array: 指定 dtype 的 JAX 数组。
    """
    return jnp.asarray(arr, dtype=getattr(jnp, dtype))


def _prepare_inputs(rwkv7_sn_inputs, head_first, dtype="bfloat16"):
    """把 rwkv7_sn_inputs 转成 JAX 测试张量。

    Args:
        rwkv7_sn_inputs: dict, 来自 fixture 的 numpy 输入。
        dtype: str, r/k/v/a/b/w 的目标 dtype。
        head_first: bool, 是否将 layout 转置为 [B, H, T, K]。

    Returns:
        tuple: (r, k, v, a, b, w, tau, mask, h0)。
            r/k/v/a/b/w: dtype 的 JAX 数组。
            tau/mask/h0: float32 的 JAX 数组。
    """
    B, T, H, _ = rwkv7_sn_inputs["r"].shape
    if head_first:
        r = _to_jax(np.transpose(rwkv7_sn_inputs["r"], (0, 2, 1, 3)), dtype)
        k = _to_jax(np.transpose(rwkv7_sn_inputs["k"], (0, 2, 1, 3)), dtype)
        v = _to_jax(np.transpose(rwkv7_sn_inputs["v"], (0, 2, 1, 3)), dtype)
        a = _to_jax(np.transpose(rwkv7_sn_inputs["a"], (0, 2, 1, 3)), dtype)
        b = _to_jax(np.transpose(rwkv7_sn_inputs["b"], (0, 2, 1, 3)), dtype)
        w = _to_jax(np.transpose(rwkv7_sn_inputs["w"], (0, 2, 1, 3)), dtype)
    else:
        r = _to_jax(rwkv7_sn_inputs["r"], dtype)
        k = _to_jax(rwkv7_sn_inputs["k"], dtype)
        v = _to_jax(rwkv7_sn_inputs["v"], dtype)
        a = _to_jax(rwkv7_sn_inputs["a"], dtype)
        b = _to_jax(rwkv7_sn_inputs["b"], dtype)
        w = _to_jax(rwkv7_sn_inputs["w"], dtype)
    tau = _to_jax(rwkv7_sn_inputs["tau"], "float32")
    mask = jnp.ones((B, T // 16), dtype=jnp.float32)
    h0 = _to_jax(rwkv7_sn_inputs["h0"], "float32")
    return r, k, v, a, b, w, tau, mask, h0


def _test_is_close(name, ref, tgt, atol, rtol, min_exact_rate=None):
    """打印精确匹配率与误差统计，再调用 assert_allclose_with_stats。

    Args:
        name: str, 比较项名称。
        ref: 参考张量，任意后端。
        tgt: 目标张量，任意后端。
        atol: float, 绝对容差。
        rtol: float, 相对容差。
        min_exact_rate: float 或 None, 仅用于打印，不参与断言。

    Returns:
        None。断言失败时抛出 AssertionError。
    """
    ref_f = np.asarray(ref, dtype=np.float32)
    tgt_f = np.asarray(tgt, dtype=np.float32)
    diff = np.abs(ref_f - tgt_f)
    total = ref_f.size
    exact_rate = np.sum(diff < 1e-7) / total * 100
    avg_err = diff.mean()
    max_diff = diff.max()
    print("-" * 80)
    print(
        f"[{name}] exact={exact_rate:.2f}%, avg_err={avg_err:.6e}, max_diff={max_diff:.6e}"
    )
    # exact match rate 仅作为诊断信息打印，不作为通过/失败条件。
    # 数值正确性由 assert_allclose_with_stats 的 atol/rtol 保证。
    assert_allclose_with_stats(ref, tgt, name, atol=atol, rtol=rtol)


@pytest.mark.jax
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_sn_forward_state(
    rwkv7_sn_jax_op, rwkv7_sn_native_op, rwkv7_sn_inputs, head_first
):
    """对比 CUDA 与 native 前向输出和最终 state（带 mask）。"""
    r_ref, k_ref, v_ref, a_ref, b_ref, w_ref, tau_ref, mask_ref, h0_ref = (
        _prepare_inputs(rwkv7_sn_inputs, head_first, "bfloat16")
    )
    r_c, k_c, v_c, a_c, b_c, w_c, tau_c, mask_c, h0_c = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "bfloat16"
    )

    y_ref, s_ref = rwkv7_sn_native_op(
        r=r_ref,
        w=w_ref,
        k=k_ref,
        v=v_ref,
        a=a_ref,
        b=b_ref,
        tau=tau_ref,
        mask=mask_ref,
        initial_state=h0_ref,
        output_final_state=True,
        head_first=head_first,
    )
    y_c, s_c = rwkv7_sn_jax_op(
        r=r_c,
        w=w_c,
        k=k_c,
        v=v_c,
        a=a_c,
        b=b_c,
        tau=tau_c,
        mask=mask_c,
        initial_state=h0_c,
        output_final_state=True,
        head_first=head_first,
    )

    _test_is_close(
        f"y_head_first={head_first}",
        y_ref,
        y_c,
        atol=1e-4,
        rtol=1e-2,
        min_exact_rate=99.0,
    )
    _test_is_close(
        f"final_state_head_first={head_first}",
        s_ref,
        s_c,
        atol=1e-5,
        rtol=1e-3,
        min_exact_rate=55.0,
    )


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_sn_backward(
    rwkv7_sn_jax_op, rwkv7_sn_native_op, rwkv7_sn_inputs, head_first
):
    """CUDA custom_vjp 反向梯度与 native Keras 实现逐元素对比（含 tau/mask）。"""
    r, k, v, a, b, w, tau, mask, h0 = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "bfloat16"
    )

    def loss(op, params):
        w, r, k, v, a, b, tau, mask, h0 = params
        y, state = op(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            tau=tau,
            mask=mask,
            initial_state=h0,
            output_final_state=True,
            head_first=head_first,
        )
        return jnp.mean(jnp.asarray(y, jnp.float32) ** 2) + jnp.mean(
            jnp.asarray(state, jnp.float32) ** 2
        )

    ref_grads = jax.grad(lambda *p: loss(rwkv7_sn_native_op, p), argnums=range(9))(
        w, r, k, v, a, b, tau, mask, h0
    )
    cuda_grads = jax.grad(lambda *p: loss(rwkv7_sn_jax_op, p), argnums=range(9))(
        w, r, k, v, a, b, tau, mask, h0
    )

    names = ["w", "r", "k", "v", "a", "b", "tau", "mask", "h0"]
    for name, g_ref, g_c in zip(names, ref_grads, cuda_grads):
        assert_allclose_with_stats(
            g_ref,
            g_c,
            f"grad_{name}_head_first={head_first}",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.jax
@pytest.mark.slow
def test_rwkv7_sn_rnn(
    rwkv7_sn_jax_op, rwkv7_sn_rnn_op, rwkv7_sn_rnn_native_op, rwkv7_sn_inputs
):
    """验证单步 RNN 与 native 单步在 16 步内一致。"""
    inputs = {k: v.copy() for k, v in rwkv7_sn_inputs.items()}
    B, T, H, K = inputs["r"].shape

    prefill_len = 16
    pre_inputs = {k: v[:, :prefill_len] for k, v in inputs.items()}
    pre_inputs["tau"] = inputs["tau"][:, : prefill_len // 16]
    r_c, k_c, v_c, a_c, b_c, w_c, tau_c, mask_c, h0_c = _prepare_inputs(
        pre_inputs, head_first=False, dtype="bfloat16"
    )

    _, state = rwkv7_sn_jax_op(
        r=r_c,
        w=w_c,
        k=k_c,
        v=v_c,
        a=a_c,
        b=b_c,
        tau=tau_c,
        mask=mask_c,
        initial_state=h0_c,
        output_final_state=True,
        head_first=False,
    )

    native_state = state
    cuda_state = state

    for step in range(prefill_len):
        rr = inputs["r"][:, prefill_len + step : prefill_len + step + 1]
        kk = inputs["k"][:, prefill_len + step : prefill_len + step + 1]
        vv = inputs["v"][:, prefill_len + step : prefill_len + step + 1]
        aa = inputs["a"][:, prefill_len + step : prefill_len + step + 1]
        bb = inputs["b"][:, prefill_len + step : prefill_len + step + 1]
        ww = inputs["w"][:, prefill_len + step : prefill_len + step + 1]
        tau = inputs["tau"][:, prefill_len // 16]
        do_sn = step == 15

        r_s = _to_jax(rr, "bfloat16")
        k_s = _to_jax(kk, "bfloat16")
        v_s = _to_jax(vv, "bfloat16")
        a_s = _to_jax(aa, "bfloat16")
        b_s = _to_jax(bb, "bfloat16")
        w_s = _to_jax(ww, "bfloat16")
        tau_s = _to_jax(tau, "float32")

        cuda_y, cuda_state = rwkv7_sn_rnn_op(
            r=r_s,
            w=w_s,
            k=k_s,
            v=v_s,
            a=a_s,
            b=b_s,
            tau=tau_s,
            do_sn=do_sn,
            initial_state=cuda_state,
            output_final_state=True,
            head_first=False,
        )

        native_y, native_state = rwkv7_sn_rnn_native_op(
            r=r_s,
            w=w_s,
            k=k_s,
            v=v_s,
            a=a_s,
            b=b_s,
            tau=tau_s,
            do_sn=do_sn,
            initial_state=native_state,
            output_final_state=True,
            head_first=False,
        )

        _test_is_close(
            f"rnn_y_step_{step}",
            native_y,
            cuda_y,
            atol=1e-4,
            rtol=1e-2,
            min_exact_rate=99.0,
        )
        # 单步 CUDA 使用 fast_math tanhf，SN 后与 native 的 ops.tanh 可能差几个 1e-5，
        # 完全一致率会下降，但 close_match/max_diff 仍能守住数值正确性。
        _test_is_close(
            f"rnn_state_step_{step}",
            native_state,
            cuda_state,
            atol=1e-5,
            rtol=1e-3,
            min_exact_rate=0.0,
        )


@pytest.mark.jax
def test_rwkv7_sn_no_mask_y_matches_all_one(rwkv7_sn_jax_op, rwkv7_sn_inputs):
    """无 mask 算子与全 1 mask 算子的 y 应一致；无 mask 路径返回 None state。"""
    B, T, H, K = rwkv7_sn_inputs["r"].shape
    n_chunks = T // 16
    mask = jnp.ones((B, n_chunks), dtype=jnp.float32)
    r, k, v, a, b, w, tau, _, h0 = _prepare_inputs(rwkv7_sn_inputs, head_first=False)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        y_no_mask, s_no_mask = rwkv7_sn_jax_op(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            tau=tau,
            initial_state=h0,
            output_final_state=True,
        )
        assert len(rec) == 1 and issubclass(rec[-1].category, UserWarning)

    y_all_one, s_all_one = rwkv7_sn_jax_op(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        tau=tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )

    assert s_no_mask is None
    pred_diff = float(jnp.max(jnp.abs(y_all_one - y_no_mask)))
    assert pred_diff < 1e-5, f"全 1 Mask 输出不一致 (max_diff={pred_diff:.3e})"
    # 带 mask 全 1 时 state 仍可正常返回
    assert s_all_one is not None


@pytest.mark.jax
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_sn_no_mask_forward_state(
    rwkv7_sn_jax_op, rwkv7_sn_native_op, rwkv7_sn_inputs, head_first
):
    """output_final_state=False 时走无 mask 算子，y 与 native 一致且不返回 state。"""
    r_ref, k_ref, v_ref, a_ref, b_ref, w_ref, tau_ref, _, h0_ref = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "bfloat16"
    )
    r_c, k_c, v_c, a_c, b_c, w_c, tau_c, _, h0_c = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "bfloat16"
    )

    y_ref = rwkv7_sn_native_op(
        r=r_ref,
        w=w_ref,
        k=k_ref,
        v=v_ref,
        a=a_ref,
        b=b_ref,
        tau=tau_ref,
        initial_state=h0_ref,
        output_final_state=False,
        head_first=head_first,
    )
    y_c = rwkv7_sn_jax_op(
        r=r_c,
        w=w_c,
        k=k_c,
        v=v_c,
        a=a_c,
        b=b_c,
        tau=tau_c,
        initial_state=h0_c,
        output_final_state=False,
        head_first=head_first,
    )

    _test_is_close(
        f"y_no_mask_head_first={head_first}",
        y_ref,
        y_c,
        atol=1e-4,
        rtol=1e-2,
        min_exact_rate=99.0,
    )


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_sn_no_mask_backward(
    rwkv7_sn_jax_op, rwkv7_sn_native_op, rwkv7_sn_inputs, head_first
):
    """无 mask 路径反向梯度与 native 对比（output_final_state=False，仅对 y 求导）。"""
    r, k, v, a, b, w, tau, _, h0 = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "bfloat16"
    )

    def loss(op, params):
        w, r, k, v, a, b, tau, h0 = params
        y = op(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            tau=tau,
            initial_state=h0,
            output_final_state=False,
            head_first=head_first,
        )
        return jnp.mean(jnp.asarray(y, jnp.float32) ** 2)

    ref_grads = jax.grad(lambda *p: loss(rwkv7_sn_native_op, p), argnums=range(8))(
        w, r, k, v, a, b, tau, h0
    )
    cuda_grads = jax.grad(lambda *p: loss(rwkv7_sn_jax_op, p), argnums=range(8))(
        w, r, k, v, a, b, tau, h0
    )

    names = ["w", "r", "k", "v", "a", "b", "tau", "h0"]
    for name, g_ref, g_c in zip(names, ref_grads, cuda_grads):
        assert_allclose_with_stats(
            g_ref,
            g_c,
            f"grad_no_mask_{name}_head_first={head_first}",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.jax
def test_rwkv7_sn_inference_arbitrary_length(
    rwkv7_sn_jax_op, rwkv7_sn_inference_op, rwkv7_sn_inputs
):
    """推理入口支持 T 不被 16 整除，此时 tau 长度只需等于 T // 16。"""
    B, T, H, K = rwkv7_sn_inputs["r"].shape
    actual_len = 34
    pre_inputs = {k: v[:, :actual_len] for k, v in rwkv7_sn_inputs.items()}
    pre_inputs["tau"] = rwkv7_sn_inputs["tau"][:, : actual_len // 16]

    r, k, v, a, b, w, tau, _, h0 = _prepare_inputs(
        pre_inputs, head_first=False, dtype="bfloat16"
    )

    # output_final_state=False 时不警告，也不返回 state。
    y = rwkv7_sn_inference_op(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        tau=tau,
        initial_state=h0,
        output_final_state=False,
        head_first=False,
    )
    assert y.shape == (B, actual_len, H, K)

    # output_final_state=True 且 mask=None 时返回 None state 并警告。
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        y2, s = rwkv7_sn_inference_op(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            tau=tau,
            initial_state=h0,
            output_final_state=True,
            head_first=False,
        )
    assert y2.shape == (B, actual_len, H, K)
    assert s is None
    assert len(rec) == 1 and issubclass(rec[-1].category, UserWarning)


@pytest.mark.jax
def test_rwkv7_sn_irregular_padding(
    rwkv7_sn_jax_op, rwkv7_sn_native_op, rwkv7_sn_rnn_native_op, rwkv7_sn_inputs
):
    """
    验证不规则长度 padding 场景：实际长度 34，pad 到 48，
    padding chunk mask=0，且 padding 位置 k=v=a=b=0, w=-inf。
    """
    B, T, H, K = rwkv7_sn_inputs["r"].shape
    actual_len = 34
    pad_len = ((actual_len + 15) // 16) * 16  # 48
    assert pad_len <= T

    def _pad(name, val, pad_val):
        full = rwkv7_sn_inputs[name][:, :pad_len].copy()
        full[:, actual_len:] = pad_val
        return full

    r = _pad("r", rwkv7_sn_inputs["r"][:, :pad_len], 0.0)
    k = _pad("k", rwkv7_sn_inputs["k"][:, :pad_len], 0.0)
    v = _pad("v", rwkv7_sn_inputs["v"][:, :pad_len], 0.0)
    a = _pad("a", rwkv7_sn_inputs["a"][:, :pad_len], 0.0)
    b = _pad("b", rwkv7_sn_inputs["b"][:, :pad_len], 0.0)
    w = _pad("w", rwkv7_sn_inputs["w"][:, :pad_len], -1e9)

    tau_full = rwkv7_sn_inputs["tau"][:, : pad_len // 16].copy()
    mask_np = np.ones((B, pad_len // 16), dtype=np.float32)
    mask_np[:, actual_len // 16 :] = 0.0

    h0 = rwkv7_sn_inputs["h0"]

    tensors = {
        "r": _to_jax(r, "bfloat16"),
        "k": _to_jax(k, "bfloat16"),
        "v": _to_jax(v, "bfloat16"),
        "a": _to_jax(a, "bfloat16"),
        "b": _to_jax(b, "bfloat16"),
        "w": _to_jax(w, "bfloat16"),
        "tau": _to_jax(tau_full, "float32"),
        "mask": _to_jax(mask_np, "float32"),
        "h0": _to_jax(h0, "float32"),
    }

    _, state_cuda = rwkv7_sn_jax_op(
        r=tensors["r"],
        w=tensors["w"],
        k=tensors["k"],
        v=tensors["v"],
        a=tensors["a"],
        b=tensors["b"],
        tau=tensors["tau"],
        mask=tensors["mask"],
        initial_state=tensors["h0"],
        output_final_state=True,
        head_first=False,
    )

    # 以训练版本跑完前 32 个 token（mask 全 1）作为参考 state。
    pre_tensors = {
        "r": _to_jax(r[:, :32], "bfloat16"),
        "k": _to_jax(k[:, :32], "bfloat16"),
        "v": _to_jax(v[:, :32], "bfloat16"),
        "a": _to_jax(a[:, :32], "bfloat16"),
        "b": _to_jax(b[:, :32], "bfloat16"),
        "w": _to_jax(w[:, :32], "bfloat16"),
        "tau": _to_jax(tau_full[:, :2], "float32"),
        "mask": jnp.ones((B, 2), dtype=jnp.float32),
        "h0": _to_jax(h0, "float32"),
    }
    _, state_ref = rwkv7_sn_native_op(
        r=pre_tensors["r"],
        w=pre_tensors["w"],
        k=pre_tensors["k"],
        v=pre_tensors["v"],
        a=pre_tensors["a"],
        b=pre_tensors["b"],
        tau=pre_tensors["tau"],
        mask=pre_tensors["mask"],
        initial_state=pre_tensors["h0"],
        output_final_state=True,
        head_first=False,
    )

    # 再用 native 单步跑 token 32,33（不触发 SN）
    for step in range(32, actual_len):
        rr = _to_jax(rwkv7_sn_inputs["r"][:, step : step + 1], "bfloat16")
        kk = _to_jax(rwkv7_sn_inputs["k"][:, step : step + 1], "bfloat16")
        vv = _to_jax(rwkv7_sn_inputs["v"][:, step : step + 1], "bfloat16")
        aa = _to_jax(rwkv7_sn_inputs["a"][:, step : step + 1], "bfloat16")
        bb = _to_jax(rwkv7_sn_inputs["b"][:, step : step + 1], "bfloat16")
        ww = _to_jax(rwkv7_sn_inputs["w"][:, step : step + 1], "bfloat16")
        tau_s = _to_jax(rwkv7_sn_inputs["tau"][:, 2], "float32")
        _, state_ref = rwkv7_sn_rnn_native_op(
            r=rr,
            w=ww,
            k=kk,
            v=vv,
            a=aa,
            b=bb,
            tau=tau_s,
            do_sn=False,
            initial_state=state_ref,
            output_final_state=True,
            head_first=False,
        )

    _test_is_close(
        "irregular_padding_state",
        state_ref,
        state_cuda,
        atol=1e-4,
        rtol=1e-3,
        min_exact_rate=5.0,
    )


@pytest.mark.jax
def test_rwkv7_sn_head_sharding(rwkv7_sn_jax_op, rwkv7_sn_inputs):
    """验证 head 维度可以沿 'h' 轴分片（TP）。单 GPU 下用 1-device mesh 模拟。"""
    devices = jax.devices()
    mesh = Mesh(devices, ("h",))

    q_spec = PartitionSpec(None, None, "h", None)
    tau_spec = PartitionSpec(None, None, "h")
    h0_spec = PartitionSpec(None, "h", None, None)
    mask_spec = PartitionSpec(None, None)

    r, k, v, a, b, w, tau, mask, h0 = _prepare_inputs(
        rwkv7_sn_inputs, head_first=False, dtype="bfloat16"
    )

    in_shardings = (
        NamedSharding(mesh, q_spec),  # r
        NamedSharding(mesh, q_spec),  # w
        NamedSharding(mesh, q_spec),  # k
        NamedSharding(mesh, q_spec),  # v
        NamedSharding(mesh, q_spec),  # a
        NamedSharding(mesh, q_spec),  # b
        NamedSharding(mesh, tau_spec),  # tau
        NamedSharding(mesh, mask_spec),  # mask
        NamedSharding(mesh, h0_spec),  # h0
    )

    def run(r, w, k, v, a, b, tau, mask, h0):
        return rwkv7_sn_jax_op(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            tau=tau,
            mask=mask,
            initial_state=h0,
            output_final_state=True,
            head_first=False,
        )

    run_sharded = jax.jit(run, in_shardings=in_shardings)
    y_ref, s_ref = run(r, w, k, v, a, b, tau, mask, h0)
    y_s, s_s = run_sharded(r, w, k, v, a, b, tau, mask, h0)

    _test_is_close("y_head_tp", y_ref, y_s, atol=1e-4, rtol=1e-2)
    _test_is_close("final_state_head_tp", s_ref, s_s, atol=1e-5, rtol=1e-3)

    # 输出 y 的 head 维度应当与输入一致沿 'h' 分片。
    assert y_s.sharding.mesh.axis_names == ("h",)
