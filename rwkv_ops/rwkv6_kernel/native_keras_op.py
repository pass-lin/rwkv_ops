"""RWKV-6 原生 Keras-ops 函数式封装。"""

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
    """RWKV-6 原生 Keras-ops 函数式算子。

    Args:
        r, k, v, w: [B, T, C]（head_first=False）或 [B, H, T, N]
            （head_first=True），bfloat16/float16/float32。
        u: [H, N] 或 [C]，与输入同 dtype。位置相关衰减项。
        initial_state: [B, H, N, N] 或 [H, N, N]，float32，可选。初始状态。
        output_final_state: bool，是否返回最终状态。
        state_map: [B] int64，可选。当 initial_state 的 batch 维度与 B
            不一致时使用。
        head_size: int，默认 64。每个 head 的维度。
        max_sequence_length: int，默认 4096。最大序列长度（原生实现不限制，
            仅用于与 CUDA 版本签名对齐）。

    Returns:
        y: [B, T, C] 或 [B, H, T, N]，与输入同 layout/dtype。
        final_state: [B, H, N, N]，float32（当 output_final_state=True）。

    Examples:
        >>> y = rwkv6(r, k, v, w, u)
        >>> y, state = rwkv6(r, k, v, w, u, initial_state=h0,
        ...                  output_final_state=True)
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
