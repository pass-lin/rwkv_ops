"""GDN / DeltaNet recurrent 家族的梯度 dtype 契约测试（PyTorch 后端）。

覆盖：反向梯度 dtype 契约（q/k/v 跟随输入、g/beta/tau/h0 固定 float32）、
上游 bf16 层直连算子时的反向可执行性、q/k/v dtype 不一致报错、
CUDA 路径非 bfloat16 输入警告并还原输出 dtype。
"""

import pytest

from rwkv_ops.delta_net_chunk import get_delta_net_chunk
from rwkv_ops.delta_net_chunk_sane import get_delta_net_chunk_sane
from rwkv_ops.delta_net_recurrent import get_delta_net_recurrent
from rwkv_ops.delta_net_recurrent_sane import get_delta_net_recurrent_sane
from rwkv_ops.gdn_chunk import get_gated_delta_net_chunk
from rwkv_ops.gdn_chunk_sane import get_gated_delta_net_chunk_sane
from rwkv_ops.gdn_recurrent import get_gated_delta_net_recurrent
from rwkv_ops.gdn_recurrent_sane import get_gated_delta_net_recurrent_sane

torch = pytest.importorskip("torch")

from tests.conftest import (  # noqa: E402
    assert_allclose_with_stats,
    assert_grad_dtypes,
    dtype_name,
)

pytestmark = pytest.mark.torch

# (id, 工厂函数, 输入 fixture 名, 是否带 g, 是否带 tau/mask)
FAMILIES = [
    ("gdn", get_gated_delta_net_recurrent, "gdn_inputs", True, False),
    ("gdn_sane", get_gated_delta_net_recurrent_sane, "gdn_sane_inputs", True, True),
    ("delta_net", get_delta_net_recurrent, "delta_net_inputs", False, False),
    (
        "delta_net_sane",
        get_delta_net_recurrent_sane,
        "delta_net_sane_inputs",
        False,
        True,
    ),
]
FAMILY_IDS = [family[0] for family in FAMILIES]

# chunk 家族的公共 API 与 recurrent 一致（同样按 KERNEL_TYPE 分发）。
CHUNK_FAMILIES = [
    ("gdn_chunk", get_gated_delta_net_chunk, "gdn_inputs", True, False),
    (
        "gdn_chunk_sane",
        get_gated_delta_net_chunk_sane,
        "gdn_sane_inputs",
        True,
        True,
    ),
    ("delta_net_chunk", get_delta_net_chunk, "delta_net_inputs", False, False),
    (
        "delta_net_chunk_sane",
        get_delta_net_chunk_sane,
        "delta_net_sane_inputs",
        False,
        True,
    ),
]
ALL_FAMILIES = FAMILIES + CHUNK_FAMILIES
ALL_FAMILY_IDS = [family[0] for family in ALL_FAMILIES]

# torch 侧的 Triton 桥接由 KERNEL_TYPE="native" 在非 CPU 平台上自动选中
# （见 rwkv_ops.utils._use_triton）；KERNEL_TYPE="triton" 在 torch 后端会静默回退
# 纯 Keras native，因此这里传 "native" 覆盖 Triton 桥接，用例 id 仍标记为 triton。
TRITON_KERNEL_TYPE = pytest.param("native", id="triton")


@pytest.fixture(scope="module")
def contract_device():
    """梯度 dtype 契约的加速路径需要 CUDA 设备。"""
    if not torch.cuda.is_available():
        pytest.skip("gradient dtype contract checks require CUDA")
    return torch.device("cuda")


def _to(arr, dtype, device):
    """把 numpy 数组转成指定 dtype 的 CUDA 张量。"""
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _make_primals(inputs, device, has_g, has_sane, qv_dtype=torch.bfloat16):
    """构造符合 dtype 契约的输入：q/k/v 为 qv_dtype，其余为 float32。"""
    primals = {
        "q": _to(inputs["q"], qv_dtype, device),
        "k": _to(inputs["k"], qv_dtype, device),
        "v": _to(inputs["v"], qv_dtype, device),
        "beta": _to(inputs["beta"], torch.float32, device),
        "h0": _to(inputs["h0"], torch.float32, device),
    }
    if has_g:
        primals["g"] = _to(inputs["g"], torch.float32, device)
    if has_sane:
        primals["tau"] = _to(inputs["tau"], torch.float32, device)
        primals["mask"] = _to(inputs["mask"], torch.float32, device)
    return primals


def _apply(op, tensors, has_g, has_sane, output_final_state=True):
    """按公共 API 的参数顺序调用算子。"""
    args = [tensors["q"], tensors["k"], tensors["v"]]
    if has_g:
        args.append(tensors["g"])
    args.append(tensors["beta"])
    if has_sane:
        args.extend([tensors["tau"], tensors["mask"]])
    return op(
        *args,
        initial_state=tensors["h0"],
        output_final_state=output_final_state,
        chunk_size=16,
    )


def _loss(op, tensors, has_g, has_sane):
    """out + state 的标量 loss。"""
    out, state = _apply(op, tensors, has_g, has_sane)
    return out.float().pow(2).mean() + state.pow(2).mean()


def _build(factory, kernel_type, inputs, device, has_g, has_sane):
    """按 KERNEL_TYPE 构造算子与契约输入。"""
    op = factory(KERNEL_TYPE=kernel_type, chunk_size=16)
    primals = _make_primals(inputs, device, has_g, has_sane)
    return op, primals


def _leaf_tensors(primals):
    """把参与求导的输入设为叶子张量（requires_grad）。"""
    names = tuple(name for name in primals if name != "mask")
    leaves = {}
    for name in names:
        tensor = primals[name].detach().requires_grad_(True)
        leaves[name] = tensor
    return names, leaves


@pytest.mark.parametrize("kernel_type", [TRITON_KERNEL_TYPE])
@pytest.mark.parametrize("family", ALL_FAMILIES, ids=ALL_FAMILY_IDS)
def test_recurrent_backward_dtype_contract(
    family, kernel_type, request, contract_device
):
    """反向梯度 dtype 契约：q/k/v 跟随输入 dtype，g/beta/tau/h0 固定 float32。"""
    fid, factory, fixture_name, has_g, has_sane = family
    inputs = request.getfixturevalue(fixture_name)
    op, primals = _build(factory, kernel_type, inputs, contract_device, has_g, has_sane)
    names, leaves = _leaf_tensors(primals)
    tensors = dict(primals, **leaves)

    _loss(op, tensors, has_g, has_sane).backward()

    grads = [leaves[name].grad for name in names]
    assert all(grad is not None for grad in grads), f"{fid}: 存在未计算的梯度"
    assert_grad_dtypes(grads, names, tensors, label=f"{fid}/{kernel_type} ")

    out, state = _apply(op, primals, has_g, has_sane)
    assert dtype_name(out.dtype) == "bfloat16", f"{fid}/{kernel_type}: out 应跟随输入"
    assert dtype_name(state.dtype) == "float32", (
        f"{fid}/{kernel_type}: state 应为 float32"
    )


@pytest.mark.parametrize("kernel_type", [TRITON_KERNEL_TYPE])
@pytest.mark.parametrize("family", ALL_FAMILIES, ids=ALL_FAMILY_IDS)
def test_recurrent_upstream_bf16_chain(family, kernel_type, request, contract_device):
    """上游 bf16 层（silu）输出直连算子时，反向链路必须可执行且梯度为 bf16。"""
    fid, factory, fixture_name, has_g, has_sane = family
    inputs = request.getfixturevalue(fixture_name)
    op, primals = _build(factory, kernel_type, inputs, contract_device, has_g, has_sane)
    v_leaf = primals["v"].detach().requires_grad_(True)
    tensors = dict(primals, v=torch.nn.functional.silu(v_leaf))

    _loss(op, tensors, has_g, has_sane).backward()

    grad = v_leaf.grad
    assert grad is not None, f"{fid}/{kernel_type}: v 的梯度为空"
    assert dtype_name(grad.dtype) == "bfloat16", (
        f"{fid}/{kernel_type}: 上游 bf16 链路的梯度应为 bfloat16，实际 {grad.dtype}"
    )


@pytest.mark.parametrize("kernel_type", [TRITON_KERNEL_TYPE])
@pytest.mark.parametrize("family", ALL_FAMILIES, ids=ALL_FAMILY_IDS)
def test_recurrent_mismatched_qkv_dtype_raises(
    family, kernel_type, request, contract_device
):
    """q/k/v dtype 不一致时必须显式报错，而不是静默 cast。"""
    fid, factory, fixture_name, has_g, has_sane = family
    inputs = request.getfixturevalue(fixture_name)
    op, primals = _build(factory, kernel_type, inputs, contract_device, has_g, has_sane)
    mixed = dict(primals)
    mixed["v"] = primals["v"].to(torch.float32)
    with pytest.raises((ValueError, TypeError)):
        _apply(op, mixed, has_g, has_sane)


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES, ids=FAMILY_IDS)
def test_recurrent_cuda_fp32_input_warns_and_restores_dtype(
    family, request, contract_device
):
    """CUDA 路径要求 q/k/v 同 dtype；全 fp32 时警告并 cast，out 按输入 dtype 还原。"""
    fid, factory, fixture_name, has_g, has_sane = family
    inputs = request.getfixturevalue(fixture_name)
    op, primals = _build(factory, "cuda", inputs, contract_device, has_g, has_sane)

    mixed = dict(primals)
    mixed["v"] = primals["v"].to(torch.float32)
    with pytest.raises((ValueError, TypeError)):
        _apply(op, mixed, has_g, has_sane)

    fp32 = _make_primals(
        inputs, contract_device, has_g, has_sane, qv_dtype=torch.float32
    )
    with pytest.warns(UserWarning):
        out_fp32, state_fp32 = _apply(op, fp32, has_g, has_sane)

    assert dtype_name(out_fp32.dtype) == "float32", (
        f"{fid}: fp32 输入应按输入 dtype 返回"
    )
    assert dtype_name(state_fp32.dtype) == "float32", f"{fid}: state 应为 float32"

    out_bf16, state_bf16 = _apply(op, primals, has_g, has_sane)
    assert_allclose_with_stats(
        out_bf16,
        out_fp32,
        f"{fid}: fp32 输入与 bf16 输入等价（out）",
        atol=1e-3,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_bf16,
        state_fp32,
        f"{fid}: fp32 输入与 bf16 输入等价（state）",
        atol=1e-3,
        rtol=1e-2,
    )
