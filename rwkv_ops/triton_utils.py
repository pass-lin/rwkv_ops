"""Triton kernel 共享工具函数。"""

import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    """数值稳定的 tanh 实现。

    Triton 标准库未提供 tanh，使用 exp 形式的分段实现：
    tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)

    Args:
        x: Triton tensor，任意形状。

    Returns:
        与 x 同形状的 tanh(x)。
    """
    x2 = 2.0 * x
    return tl.where(
        x >= 0.0,
        1.0 - 2.0 / (tl.exp(x2) + 1.0),
        2.0 / (tl.exp(-x2) + 1.0) - 1.0,
    )


@triton.jit
def _sech2(x):
    """tanh 的导数 sech^2(x) = 1 - tanh^2(x)。

    Args:
        x: Triton tensor。

    Returns:
        与 x 同形状的 sech^2(x)。
    """
    t = _tanh(x)
    return 1.0 - t * t


@triton.jit
def _sane_transform(state, tau):
    """State Anomaly Neutralization 前向变换：tau * tanh(state / tau)。

    Args:
        state: Triton tensor。
        tau: 标量或与 state 广播的 tensor，> 0。

    Returns:
        与 state 同形状的 SANE 后状态。
    """
    tau_safe = tl.maximum(tau, 1e-6)
    return tau_safe * _tanh(state / tau_safe)


@triton.jit
def _sane_backward_factor(s_raw, tau):
    """SANE 反向的局部导数因子。

    给定 SANE 前状态 s_raw 和阈值 tau，返回
        d(s_sane) / d(s_raw) = sech^2(s_raw / tau)。

    Args:
        s_raw: SANE 前状态。
        tau: 阈值，> 0。

    Returns:
        sech^2(s_raw / tau)。
    """
    tau_safe = tl.maximum(tau, 1e-6)
    return _sech2(s_raw / tau_safe)


@triton.jit
def _sane_dtau_factor(d_out, s_raw, tau):
    """SANE 对 tau 的梯度贡献。

    d/dtau [tau * tanh(s/tau)] = tanh(s/tau) - (s/tau) * sech^2(s/tau)。

    Args:
        d_out: 上游梯度。
        s_raw: SANE 前状态。
        tau: 阈值，> 0。

    Returns:
        d_out * (tanh(s/tau) - (s/tau) * sech^2(s/tau))。
    """
    tau_safe = tl.maximum(tau, 1e-6)
    u = s_raw / tau_safe
    t = _tanh(u)
    sech2 = 1.0 - t * t
    return d_out * (t - u * sech2)
