"""Gated DeltaNet recurrent Triton kernel（共享实现）。"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def gated_delta_net_recurrent_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    kv_mem_out,
    state_chkp,
    inv_norm_q,
    inv_norm_k,
    h0,
    ht,
    scale,
    B,
    H,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """Gated DeltaNet recurrent 训练前向 Triton kernel。

    每个 program 处理一个 (batch, head) 与 V 方向的一个 block。
    除 out 与最终 state 外，还保存反向所需的 kv_mem 与 chunk 级 state 快照。

    Args:
        q, k: [B, H, T, K]，与输入同 dtype，row-major。
        v, o: [B, H, T, V]，v 与 o 同 dtype，row-major。
        kv_mem_out: [B, H, T, V]，float32，每步的 `k_t @ state_{t-1}`。
        state_chkp: [B, H, T//CHUNK_LEN, K, V]，float32，chunk 末尾 state 快照。
        inv_norm_q: [B, H, T, 1]，float32，q 的 L2 归一化逆范数。
        inv_norm_k: [B, H, T, 1]，float32，k 的 L2 归一化逆范数。
        g: [B, H, T]，log-space decay，row-major。
        beta: [B, H, T]，已在外部过 sigmoid 的写入强度，row-major。
        h0: [B, H, K, V]，float32，初始 state。
        ht: [B, H, K, V]，float32，最终 state（仅当 STORE_FINAL_STATE=True 时写入）。
        scale: float，query 缩放系数。
        B, H, T: int，batch/head/sequence 大小。
        K, V: tl.constexpr，head 维度大小。
        BK, BV: tl.constexpr，block 大小。
        CHUNK_LEN: tl.constexpr，chunk 长度，固定 16。
    """
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_b = i_nh // H
    i_h = i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    i_b_i64 = i_b.to(tl.int64)
    i_h_i64 = i_h.to(tl.int64)
    H_i64 = tl.cast(H, tl.int64)
    T_i64 = tl.cast(T, tl.int64)
    K_i64 = tl.cast(K, tl.int64)
    V_i64 = tl.cast(V, tl.int64)
    CHUNK_LEN_i64 = tl.cast(CHUNK_LEN, tl.int64)

    bh = i_b_i64 * H_i64 + i_h_i64

    base_qk = bh * T_i64 * K_i64 + o_k
    base_vo = bh * T_i64 * V_i64 + o_v
    base_gb = bh * T_i64
    base_inv_norm = bh * T_i64
    base_state = bh * K_i64 * V_i64 + o_k[:, None] * V_i64 + o_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + base_state, mask=mask_h, other=0.0).to(tl.float32)

    num_chunks = T_i64 // CHUNK_LEN_i64

    for t in range(T):
        t_i64 = t.to(tl.int64)

        p_q = q + base_qk + t_i64 * K_i64
        p_k = k + base_qk + t_i64 * K_i64
        p_v = v + base_vo + t_i64 * V_i64
        p_g = g + base_gb + t_i64
        p_beta = beta + base_gb + t_i64
        p_o = o + base_vo + t_i64 * V_i64
        p_kv_mem = bh * T_i64 * V_i64 + t_i64 * V_i64 + o_v

        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)

        # L2 归一化，先算逆范数保存供反向使用，再对 q/k 归一化。
        inv_norm_q_t = tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        inv_norm_k_t = tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        tl.store(inv_norm_q + base_inv_norm + t_i64, inv_norm_q_t)
        tl.store(inv_norm_k + base_inv_norm + t_i64, inv_norm_k_t)

        b_q = b_q * inv_norm_q_t * scale
        b_k = b_k * inv_norm_k_t

        # log-space decay。
        b_h = b_h * tl.exp(b_g)

        # kv_mem = sum_K(state * k)，供反向使用。
        kv_mem = tl.sum(b_h * b_k[:, None], axis=0)
        tl.store(
            kv_mem_out + p_kv_mem,
            kv_mem.to(kv_mem_out.dtype.element_ty),
            mask=mask_v,
        )

        # delta = beta * (v - kv_mem)。
        b_v = b_beta * (b_v - kv_mem)

        # state += k^T delta。
        b_h = b_h + b_k[:, None] * b_v[None, :]

        # out = sum_K(state * q)。
        b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(p_o, b_o.to(o.dtype.element_ty), mask=mask_v)

        # 每 chunk 末尾保存 state 快照，供反向重算使用。
        if num_chunks > 0 and (t + 1) % CHUNK_LEN == 0:
            chkp_t = (t_i64 + 1) // CHUNK_LEN_i64 - 1
            base_chkp = (
                bh * num_chunks * K_i64 * V_i64
                + chkp_t * K_i64 * V_i64
                + o_k[:, None] * V_i64
                + o_v[None, :]
            )
            tl.store(
                state_chkp + base_chkp,
                b_h.to(state_chkp.dtype.element_ty),
                mask=mask_h,
            )

    if STORE_FINAL_STATE:
        tl.store(ht + base_state, b_h.to(ht.dtype.element_ty), mask=mask_h)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def gated_delta_net_recurrent_bwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    do,
    dht,
    kv_mem_out,
    inv_norm_q,
    inv_norm_k,
    h0,
    state_chkp,
    dq,
    dk,
    dv,
    dg,
    dbeta,
    dh0,
    scale,
    B,
    H,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
):
    """Gated DeltaNet recurrent 训练反向 Triton kernel。

    每个 program 处理一个 (batch, head) 与 V 方向的一个 block。
    先重新跑一遍前向得到最终状态，再从 T 反向递推到 0。
    q/k/g/beta 梯度在 V 方向需要跨 block 归约，NV>1 时使用 atomic_add。

    Args:
        q, k: [B, H, T, K]，输入同 dtype，row-major。
        v, do: [B, H, T, V]，输入/输出梯度同 dtype，row-major。
        g, beta: [B, H, T]，row-major。
        dht: [B, H, K, V]，float32，最终 state 梯度。
        kv_mem_out: [B, H, T, V]，float32，前向保存的 kv_mem。
        inv_norm_q, inv_norm_k: [B, H, T]，float32，前向保存的逆范数。
        h0: [B, H, K, V]，float32，初始 state。
        dq, dk: [B, H, T, K]，float32，输出梯度。
        dv: [B, H, T, V]，float32，输出梯度。
        dg, dbeta: [B, H, T]，float32，输出梯度。
        dh0: [B, H, K, V]，float32，初始 state 梯度。
        scale: float，query 缩放系数。
        B, H, T: int。
        K, V, BK, BV, CHUNK_LEN: tl.constexpr。
        USE_FINAL_STATE_GRADIENT: 是否使用 dht。
    """
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_b = i_nh // H
    i_h = i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    i_b_i64 = i_b.to(tl.int64)
    i_h_i64 = i_h.to(tl.int64)
    H_i64 = tl.cast(H, tl.int64)
    T_i64 = tl.cast(T, tl.int64)
    K_i64 = tl.cast(K, tl.int64)
    V_i64 = tl.cast(V, tl.int64)

    bh = i_b_i64 * H_i64 + i_h_i64
    base_qk = bh * T_i64 * K_i64 + o_k
    base_vo = bh * T_i64 * V_i64 + o_v
    base_gb = bh * T_i64
    base_inv_norm = bh * T_i64
    base_state = bh * K_i64 * V_i64 + o_k[:, None] * V_i64 + o_v[None, :]

    # 前向重算得到 S_T。
    b_h = tl.load(h0 + base_state, mask=mask_h, other=0.0).to(tl.float32)
    for t in range(T):
        t_i64 = t.to(tl.int64)
        p_q = q + base_qk + t_i64 * K_i64
        p_k = k + base_qk + t_i64 * K_i64
        p_v = v + base_vo + t_i64 * V_i64
        p_g = g + base_gb + t_i64
        p_beta = beta + base_gb + t_i64

        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)

        inv_norm_q_t = tl.load(inv_norm_q + base_inv_norm + t_i64)
        inv_norm_k_t = tl.load(inv_norm_k + base_inv_norm + t_i64)

        b_q = b_q * inv_norm_q_t * scale
        b_k = b_k * inv_norm_k_t

        b_h = b_h * tl.exp(b_g)
        kv_mem = tl.sum(b_h * b_k[:, None], axis=0)
        b_v = b_beta * (b_v - kv_mem)
        b_h = b_h + b_k[:, None] * b_v[None, :]

    # 初始化状态反向传播量。
    carry_dS = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        carry_dS += tl.load(dht + base_state, mask=mask_h, other=0.0).to(tl.float32)

    CHUNK_LEN_i64 = tl.cast(CHUNK_LEN, tl.int64)
    num_chunks = T_i64 // CHUNK_LEN_i64

    # 从 T-1 反向递推到 0。
    for t in range(T - 1, -1, -1):
        t_i64 = t.to(tl.int64)

        # 到达 chunk 边界时从快照重新加载 S_{t+1}，避免长序列反向递推中
        # 反复除以 exp(g) 导致的数值爆炸。
        tp1 = t + 1
        if tp1 > 0 and tp1 % CHUNK_LEN == 0 and tp1 < T:
            chkp_t = (tp1 // CHUNK_LEN) - 1
            base_chkp = (
                bh * num_chunks * K_i64 * V_i64
                + chkp_t * K_i64 * V_i64
                + o_k[:, None] * V_i64
                + o_v[None, :]
            )
            b_h = tl.load(state_chkp + base_chkp, mask=mask_h, other=0.0).to(tl.float32)

        p_q = q + base_qk + t_i64 * K_i64
        p_k = k + base_qk + t_i64 * K_i64
        p_v = v + base_vo + t_i64 * V_i64
        p_g = g + base_gb + t_i64
        p_beta = beta + base_gb + t_i64
        p_do = do + base_vo + t_i64 * V_i64
        p_kv_mem = bh * T_i64 * V_i64 + t_i64 * V_i64 + o_v

        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0.0).to(tl.float32)

        inv_norm_q_t = tl.load(inv_norm_q + base_inv_norm + t_i64)
        inv_norm_k_t = tl.load(inv_norm_k + base_inv_norm + t_i64)

        q_hat = b_q * inv_norm_q_t
        k_hat = b_k * inv_norm_k_t
        q_tilde = q_hat * scale

        kv_mem = tl.load(kv_mem_out + p_kv_mem, mask=mask_v, other=0.0).to(tl.float32)
        delta = b_beta * (b_v - kv_mem)

        state_new = b_h
        dstate_new = carry_dS + q_tilde[:, None] * b_do[None, :]

        # d_delta = dstate_new^T @ k_hat
        d_delta = tl.sum(dstate_new * k_hat[:, None], axis=0)

        d_v = b_beta * d_delta
        d_beta = tl.sum((b_v - kv_mem) * d_delta)
        d_kv_mem = -b_beta * d_delta

        state_decay = state_new - k_hat[:, None] * delta[None, :]

        d_k_hat = tl.sum(dstate_new * delta[None, :], axis=1) + tl.sum(
            state_decay * d_kv_mem[None, :], axis=1
        )
        dstate_decay = dstate_new + k_hat[:, None] * d_kv_mem[None, :]

        exp_g = tl.exp(b_g)
        state_old = state_decay / exp_g
        d_g = tl.sum(state_decay * dstate_decay)
        carry_dS = exp_g * dstate_decay
        b_h = state_old

        d_q_tilde = tl.sum(state_new * b_do[None, :], axis=1)
        d_q_hat = scale * d_q_tilde
        q_hat_dot = tl.sum(q_hat * d_q_hat)
        d_q = inv_norm_q_t * (d_q_hat - q_hat * q_hat_dot)

        k_hat_dot = tl.sum(k_hat * d_k_hat)
        d_k = inv_norm_k_t * (d_k_hat - k_hat * k_hat_dot)

        tl.store(dv + base_vo + t_i64 * V_i64, d_v, mask=mask_v)

        if NV > 1:
            tl.atomic_add(dq + base_qk + t_i64 * K_i64, d_q, mask=mask_k)
            tl.atomic_add(dk + base_qk + t_i64 * K_i64, d_k, mask=mask_k)
            tl.atomic_add(dg + base_gb + t_i64, d_g)
            tl.atomic_add(dbeta + base_gb + t_i64, d_beta)
        else:
            tl.store(dq + base_qk + t_i64 * K_i64, d_q, mask=mask_k)
            tl.store(dk + base_qk + t_i64 * K_i64, d_k, mask=mask_k)
            tl.store(dg + base_gb + t_i64, d_g)
            tl.store(dbeta + base_gb + t_i64, d_beta)

    tl.store(dh0 + base_state, carry_dS, mask=mask_h)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def gated_delta_net_recurrent_inference_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    scale,
    B,
    H,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """Gated DeltaNet recurrent 推理前向 Triton kernel。

    与训练前向数学一致，但不输出 kv_mem 与 state_chkp，减少推理显存占用。
    每个 program 处理一个 (batch, head) 与 V 方向的一个 block。

    Args:
        q, k: [B, H, T, K]，与输入同 dtype，row-major。
        v, o: [B, H, T, V]，v 与 o 同 dtype，row-major。
        g: [B, H, T]，log-space decay，row-major。
        beta: [B, H, T]，已在外部过 sigmoid 的写入强度，row-major。
        h0: [B, H, K, V]，float32，初始 state。
        ht: [B, H, K, V]，float32，最终 state（仅当 STORE_FINAL_STATE=True 时写入）。
        scale: float，query 缩放系数。
        B, H, T: int，batch/head/sequence 大小。
        K, V: tl.constexpr，head 维度大小。
        BK, BV: tl.constexpr，block 大小。
    """
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_b = i_nh // H
    i_h = i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    i_b_i64 = i_b.to(tl.int64)
    i_h_i64 = i_h.to(tl.int64)
    H_i64 = tl.cast(H, tl.int64)
    T_i64 = tl.cast(T, tl.int64)
    K_i64 = tl.cast(K, tl.int64)
    V_i64 = tl.cast(V, tl.int64)

    bh = i_b_i64 * H_i64 + i_h_i64

    base_qk = bh * T_i64 * K_i64 + o_k
    base_vo = bh * T_i64 * V_i64 + o_v
    base_gb = bh * T_i64
    base_state = bh * K_i64 * V_i64 + o_k[:, None] * V_i64 + o_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + base_state, mask=mask_h, other=0.0).to(tl.float32)

    for t in range(T):
        t_i64 = t.to(tl.int64)

        p_q = q + base_qk + t_i64 * K_i64
        p_k = k + base_qk + t_i64 * K_i64
        p_v = v + base_vo + t_i64 * V_i64
        p_g = g + base_gb + t_i64
        p_beta = beta + base_gb + t_i64
        p_o = o + base_vo + t_i64 * V_i64

        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)

        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        b_h = b_h * tl.exp(b_g)

        kv_mem = tl.sum(b_h * b_k[:, None], axis=0)
        b_v = b_beta * (b_v - kv_mem)
        b_h = b_h + b_k[:, None] * b_v[None, :]

        b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(p_o, b_o.to(o.dtype.element_ty), mask=mask_v)

    if STORE_FINAL_STATE:
        tl.store(ht + base_state, b_h.to(ht.dtype.element_ty), mask=mask_h)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def gated_delta_net_recurrent_single_step_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    scale,
    B,
    H,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """Gated DeltaNet recurrent 单步 RNN 前向 Triton kernel。

    每个 program 处理一个 (batch, head) 与 V 方向的一个 block。
    输入没有时间维：q/k 为 [B, H, K]，v/o 为 [B, H, V]，g/beta 为 [B, H]。

    Args:
        q, k: [B, H, K]，与输入同 dtype，row-major。
        v, o: [B, H, V]，v 与 o 同 dtype，row-major。
        g: [B, H]，log-space decay，row-major。
        beta: [B, H]，已在外部过 sigmoid 的写入强度，row-major。
        h0: [B, H, K, V]，float32，初始 state。
        ht: [B, H, K, V]，float32，下一步 state（仅当 STORE_FINAL_STATE=True 时写入）。
        scale: float，query 缩放系数。
        B, H: int，batch/head 大小。
        K, V: tl.constexpr，head 维度大小。
        BK, BV: tl.constexpr，block 大小。
    """
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_b = i_nh // H
    i_h = i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    i_b_i64 = i_b.to(tl.int64)
    i_h_i64 = i_h.to(tl.int64)
    H_i64 = tl.cast(H, tl.int64)
    K_i64 = tl.cast(K, tl.int64)
    V_i64 = tl.cast(V, tl.int64)

    bh = i_b_i64 * H_i64 + i_h_i64

    base_qk = bh * K_i64 + o_k
    base_vo = bh * V_i64 + o_v
    base_state = bh * K_i64 * V_i64 + o_k[:, None] * V_i64 + o_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + base_state, mask=mask_h, other=0.0).to(tl.float32)

    b_q = tl.load(q + base_qk, mask=mask_k, other=0.0).to(tl.float32)
    b_k = tl.load(k + base_qk, mask=mask_k, other=0.0).to(tl.float32)
    b_v = tl.load(v + base_vo, mask=mask_v, other=0.0).to(tl.float32)
    b_g = tl.load(g + bh).to(tl.float32)
    b_beta = tl.load(beta + bh).to(tl.float32)

    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    b_h = b_h * tl.exp(b_g)
    kv_mem = tl.sum(b_h * b_k[:, None], axis=0)
    b_v = b_beta * (b_v - kv_mem)
    b_h = b_h + b_k[:, None] * b_v[None, :]
    b_o = tl.sum(b_h * b_q[:, None], axis=0)
    tl.store(o + base_vo, b_o.to(o.dtype.element_ty), mask=mask_v)

    if STORE_FINAL_STATE:
        tl.store(ht + base_state, b_h.to(ht.dtype.element_ty), mask=mask_h)
