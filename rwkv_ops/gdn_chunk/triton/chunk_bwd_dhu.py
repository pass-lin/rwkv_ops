"""Gated DeltaNet chunkwise 状态反向扫描 Triton kernel。

从 `dht` 和 `do` 反向递推计算 chunk 级状态梯度 `dh`、初始状态梯度 `dh0`
以及累加后的 `dv`。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [32, 64]
        for num_warps in [2, 4]
        for num_stages in [2]
    ],
    key=["K", "V", "C"],
)
@triton.jit
def _gdn_chunk_bwd_dhu_kernel(
    q_ptr,
    k_ptr,
    w_ptr,
    g_ptr,
    do_ptr,
    dv_local_ptr,
    dht_ptr,
    B,
    H,
    T,
    K,
    V,
    scale,
    dh_ptr,
    dh0_ptr,
    dv_out_ptr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """状态反向扫描 kernel。

    每个 program 处理一个 (batch, head, V-block)，沿 chunk 维反向扫描。
    固定读取 dht；调用方在未提供时传零张量，避免 USE_DHT constexpr 造成
    Triton cache/specialization 串扰。

    Args:
        q_ptr, k_ptr, w_ptr: [B, H, T, K]。
        g_ptr: [B, H, T]，cumsum 后的 log-space decay。
        do_ptr: [B, H, T, V]。
        dv_local_ptr: [B, H, T, V]，来自 chunk_bwd_dv_local 的局部 dv。
        dht_ptr: [B, H, K, V]，最终状态梯度；未提供时为全零。
        B, H, T, K, V: 维度。
        scale: float，`1/sqrt(K)`。
        C: chunk 长度，编译期常量。
        BV: V 维 block 大小。
        dh_ptr: [B, H, T//C, K, V]，输出 chunk 级状态梯度。
        dh0_ptr: [B, H, K, V]，输出初始状态梯度。
        dv_out_ptr: [B, H, T, V]，输出累加后的 dv。
    """
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    h = i_nh % H
    b = i_nh // H

    N = T // C

    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, BK)
    m_v = o_v < V
    m_h = (o_k[:, None] < K) & (o_v[None, :] < V)

    base_bh = (b * H + h).to(tl.int64)
    base_qk = base_bh * T
    base_v = base_bh * T
    base_gb = base_bh * T
    base_dh = base_bh * N

    # 初始化 dh；未提供 dht 时传零张量，结果等价于零初始化。
    p_dht = dht_ptr + base_bh * K * V + o_k[:, None] * V + o_v[None, :]
    b_dh = tl.load(p_dht, mask=m_h, other=0.0).to(tl.float32)

    scale_f32 = tl.cast(scale, tl.float32)

    for i_t in range(N - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)

        # 保存 dh_i
        p_dh = dh_ptr + (base_dh + i_t_int64) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh, b_dh.to(dh_ptr.dtype.element_ty), mask=m_h)

        o_c = tl.arange(0, C)
        m_c = o_c < C

        # 加载 q, k, w, g, do, dv_local
        base_chunk = base_qk + i_t_int64 * C
        base_chunk_v = base_v + i_t_int64 * C
        base_chunk_gb = base_gb + i_t_int64 * C

        # dv = k @ dh，沿 K 维累加
        b_dv = tl.zeros([C, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K

            p_k = k_ptr + (base_chunk * K + o_c[:, None] * K + o_k[None, :]).to(
                tl.int64
            )
            b_k = tl.load(p_k, mask=m_c[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )

            p_dh_block = (
                dh_ptr + (base_dh + i_t_int64) * K * V + o_k[:, None] * V + o_v[None, :]
            )
            b_dh_block = tl.load(
                p_dh_block, mask=m_k[:, None] & m_v[None, :], other=0.0
            ).to(tl.float32)

            b_dv += tl.dot(b_k, b_dh_block.to(b_k.dtype), allow_tf32=False)

        # 加载 g
        p_g = g_ptr + base_chunk_gb + o_c
        b_g = tl.load(p_g, mask=m_c, other=0.0).to(tl.float32)
        g_last = tl.load(g_ptr + base_chunk_gb + C - 1).to(tl.float32)

        # dv 乘以 decay：exp(g_last - g)
        b_dv *= tl.exp(g_last - b_g)[:, None]

        # 加上 dv_local 并保存最终 dv
        p_dv_local = dv_local_ptr + (
            base_chunk_v * V + o_c[:, None] * V + o_v[None, :]
        ).to(tl.int64)
        b_dv += tl.load(p_dv_local, mask=m_c[:, None] & m_v[None, :], other=0.0).to(
            tl.float32
        )
        p_dv_out = dv_out_ptr + (base_chunk_v * V + o_c[:, None] * V + o_v[None, :]).to(
            tl.int64
        )
        tl.store(
            p_dv_out,
            b_dv.to(dv_out_ptr.dtype.element_ty),
            mask=m_c[:, None] & m_v[None, :],
        )

        # 更新 dh：dh = dh * exp(g_last) + q * exp(g) @ do * scale - w @ dv
        b_dh *= tl.exp(g_last)

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K

            p_q = q_ptr + (base_chunk * K + o_c[:, None] * K + o_k[None, :]).to(
                tl.int64
            )
            b_q = tl.load(p_q, mask=m_c[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )
            b_q *= tl.exp(b_g)[:, None]

            p_do = do_ptr + (base_chunk_v * V + o_c[:, None] * V + o_v[None, :]).to(
                tl.int64
            )
            b_do = tl.load(p_do, mask=m_c[:, None] & m_v[None, :], other=0.0).to(
                tl.float32
            )

            # [C, K]^T @ [C, V] -> [K, V]，即 q^T @ do
            b_dh += (
                tl.dot(tl.trans(b_q).to(b_do.dtype), b_do, allow_tf32=False) * scale_f32
            )

            p_w = w_ptr + (base_chunk * K + o_c[:, None] * K + o_k[None, :]).to(
                tl.int64
            )
            b_w = tl.load(p_w, mask=m_c[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )

            # [C, K]^T @ [C, V] -> [K, V]，即 -w^T @ dv
            b_dh -= tl.dot(tl.trans(b_w).to(b_dv.dtype), b_dv, allow_tf32=False)

    # 保存 dh0
    p_dh0 = dh0_ptr + base_bh * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(p_dh0, b_dh.to(dh0_ptr.dtype.element_ty), mask=m_h)


def gdn_chunk_bwd_dhu(q, k, w, g, do, dv_local, dht=None, scale=1.0, chunk_size=64):
    """状态反向扫描封装。

    Args:
        q: [B, H, T, K]。
        k: [B, H, T, K]。
        w: [B, H, T, K]。
        g: [B, H, T]，cumsum 后的 log-space decay。
        do: [B, H, T, V]。
        dv_local: [B, H, T, V]。
        dht: [B, H, K, V]，可选。
        scale: float。
        chunk_size: int。

    Returns:
        dh: [B, H, T//C, K, V]。
        dh0: [B, H, K, V]。
        dv: [B, H, T, V]。
    """
    B, H, T, K = q.shape
    V = do.shape[-1]
    C = chunk_size
    N = T // C

    dh = torch.empty(B, H, N, K, V, dtype=torch.float32, device=q.device)
    dv = torch.empty_like(do)
    dh0 = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)

    BK = triton.next_power_of_2(K)

    def grid(meta):
        return (B * H * triton.cdiv(V, meta["BV"]),)

    if dht is None:
        dht = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)

    _gdn_chunk_bwd_dhu_kernel[grid](
        q,
        k,
        w,
        g,
        do,
        dv_local,
        dht,
        B,
        H,
        T,
        K,
        V,
        scale,
        dh,
        dh0,
        dv,
        C=C,
        BK=BK,
    )
    return dh, dh0, dv
