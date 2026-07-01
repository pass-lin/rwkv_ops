"""
RWKV-6 原生 Keras-ops 实现（函数式接口）。

直接复用 ops_rwkv_kernel.RWKVKernelOperator 作为数值 ground truth，
仅将其封装为与 CUDA 版本一致的函数签名。
"""

from .ops_rwkv_kernel import RWKVKernelOperator


def rwkv6(
    r,
    k,
    v,
    w,
    u,
    initial_state=None,
    output_final_state: bool = False,
    state_map=None,
    head_size: int = 64,
    max_sequence_length: int = 4096,
):
    """
    RWKV-6 原生 Keras-ops 函数式算子。

    Args:
        r, k, v, w: [B, T, C]（head_first=False）或 [B, H, T, N]（head_first=True）
        u: [H, N] 或 [C]
        initial_state: 可选，[B, H, N, N] 或 [H, N, N]
        output_final_state: 是否返回最终状态
        state_map: [B] int64，当 initial_state 的 batch 维度与 B 不一致时使用
        head_size: 每个 head 的维度，默认 64
        max_sequence_length: 最大序列长度，默认 4096

    Returns:
        y: [B, T, C]
        final_state: [B, H, N, N]（当 output_final_state=True）
    """
    op = RWKVKernelOperator(
        head_size=head_size, max_sequence_length=max_sequence_length
    )
    y, final_state = op(
        r,
        k,
        v,
        w,
        u,
        with_state=output_final_state,
        init_state=initial_state,
        state_map=state_map,
    )
    if output_final_state:
        return y, final_state
    return y
