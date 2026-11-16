"""recurrent Triton kernel 在 JAX 侧 launch 参数的回归测试。"""

import importlib
import inspect
from types import SimpleNamespace

import pytest

from rwkv_ops import triton_launch

pytest.importorskip("jax_triton")

_BRIDGE_MODULES = (
    "rwkv_ops.gdn_recurrent.jax_triton_kernel",
    "rwkv_ops.gdn_recurrent_sane.jax_triton_kernel",
    "rwkv_ops.delta_net_recurrent.jax_triton_kernel",
    "rwkv_ops.delta_net_recurrent_sane.jax_triton_kernel",
    "rwkv_ops.rwkv7_kernel.jax_triton_kernel",
    "rwkv_ops.rwkv7_sane_kernel.jax_triton_kernel",
)


class _FakeAutotuner:
    """模拟 triton.Autotuner：同时带 configs 与 fn；config 可携带 constexpr 取值。"""

    def __init__(self, fn, config_kwargs=None):
        first = SimpleNamespace(kwargs=dict(config_kwargs or {}))
        self.configs = [first, SimpleNamespace(kwargs={})]
        self.fn = fn


def _fake_kernel(*args, **kwargs):
    """占位 kernel，仅用于确认 triton_call 收到的是裸 kernel。"""


@pytest.mark.jax
def test_triton_call_unwraps_autotuner(monkeypatch):
    """Autotuner 必须被拆成裸 kernel，并补齐固定 launch 参数。"""
    import jax_triton

    recorded = {}

    def fake_triton_call(*args, **kwargs):
        recorded.update(kwargs)
        recorded["args"] = args
        return None

    monkeypatch.setattr(jax_triton, "triton_call", fake_triton_call)
    triton_launch.triton_call(
        7,
        kernel=_FakeAutotuner(_fake_kernel, {"MINI_BSZ": 1}),
        out_shape="o",
        grid=(1,),
    )

    assert recorded["kernel"] is _fake_kernel
    assert recorded["num_warps"] == triton_launch.NUM_WARPS
    assert recorded["num_stages"] == triton_launch.NUM_STAGES
    assert recorded["out_shape"] == "o"
    assert recorded["grid"] == (1,)
    assert recorded["args"] == (7,)
    assert recorded["MINI_BSZ"] == 1


@pytest.mark.jax
def test_triton_call_keeps_explicit_constexpr_over_config(monkeypatch):
    """调用方显式给出的 constexpr 优先于 autotune config 里的同名取值。"""
    import jax_triton

    recorded = {}

    def fake_triton_call(*args, **kwargs):
        recorded.update(kwargs)
        return None

    monkeypatch.setattr(jax_triton, "triton_call", fake_triton_call)
    triton_launch.triton_call(
        1,
        kernel=_FakeAutotuner(_fake_kernel, {"MINI_BSZ": 1}),
        out_shape="o",
        grid=(1,),
        MINI_BSZ=4,
    )

    assert recorded["MINI_BSZ"] == 4


@pytest.mark.jax
def test_triton_call_keeps_plain_kernel(monkeypatch):
    """不带 configs 的裸 kernel 原样透传，不依赖 fn 属性识别。"""
    import jax_triton

    recorded = {}

    def fake_triton_call(*args, **kwargs):
        recorded.update(kwargs)
        return None

    monkeypatch.setattr(jax_triton, "triton_call", fake_triton_call)
    triton_launch.triton_call(1, kernel=_fake_kernel, out_shape="o", grid=(1,))

    assert recorded["kernel"] is _fake_kernel


@pytest.mark.jax
def test_triton_call_allows_explicit_launch_override(monkeypatch):
    """调用方显式给出的 num_warps/num_stages 优先于模块默认值。"""
    import jax_triton

    recorded = {}

    def fake_triton_call(*args, **kwargs):
        recorded.update(kwargs)
        return None

    monkeypatch.setattr(jax_triton, "triton_call", fake_triton_call)
    triton_launch.triton_call(
        1, kernel=_fake_kernel, out_shape="o", grid=(1,), num_warps=8, num_stages=1
    )

    assert recorded["num_warps"] == 8
    assert recorded["num_stages"] == 1


@pytest.mark.jax
@pytest.mark.parametrize("module_name", _BRIDGE_MODULES)
def test_bridge_uses_launch_helper(module_name):
    """所有 recurrent JAX 桥接都经 triton_launch 调用，不再直连 jax_triton。"""
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)

    assert "from ..triton_launch import" in source
    assert "jt.triton_call(" not in source


@pytest.mark.jax
@pytest.mark.parametrize(
    "module_name", ("rwkv_ops.gdn_recurrent", "rwkv_ops.delta_net_recurrent")
)
def test_compute_bv_bounded_by_max_bv(module_name):
    """_compute_bv 用 RWKV_OPS_TRITON_MAX_BV 封顶，且极小 V 不被放大。"""
    kernel_module = importlib.import_module(f"{module_name}.jax_triton_kernel")
    compute_bv = getattr(kernel_module, "_compute_bv")

    assert compute_bv(256) == triton_launch.MAX_BV
    assert compute_bv(triton_launch.MAX_BV) == triton_launch.MAX_BV
    assert compute_bv(16) == 16
