"""RWKV-7-SANE 共享 Triton kernel。"""

import triton
import triton.language as tl


#  无 mask 前向（chunk 边界无条件 SANE）
@triton.autotune(
    configs=[
        triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def rwkv7_sane_fwd_kernel(
    R,
    W,
    K,
    V,
    A,
    B_param,
    TAU,
    H0,
    B_BATCH,
    N_HEAD,
    T_LEN,
    OUT,
    SA_OUT,
    STATE_CHKP,
    H_SIZE: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    MINI_BSZ: tl.constexpr,
):
    """RWKV-7-SANE 训练前向 Triton kernel（无 mask）。

    每个 block 处理一个 (batch, head)，顺序扫描 T 步，在每个 chunk 末尾
    写出 SANE 之前的 state checkpoint。

    Args:
        R, W, K, V, A, B_param: [B, H, T, K], bfloat16, row-major。
        TAU: [B, H, T//16], float32, row-major。
        H0: [B, H, K, K], float32, row-major。初始 state。
        B_BATCH, N_HEAD, T_LEN: int。batch / head / time 大小。
        OUT: [B, H, T, K], bfloat16, row-major。输出 y。
        SA_OUT: [B, H, T, K], float32, row-major。反向所需中间量。
        STATE_CHKP: [B, H, T//16, K, K], float32, row-major。SANE 之前的 state checkpoint。

    编译期宏:
        H_SIZE: head_size，必须被 4 整除。
        CHUNK_LEN: chunk 长度，固定 16。
        MINI_BSZ: 当前固定为 1，限制单 block 寄存器占用。

    指针算术一律使用 64 位整数，防止大 tensor 时 32 位偏移溢出。
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
    b_mask = b_range < B_BATCH
    cols = tl.arange(0, H_SIZE)

    b_range_i64 = b_range.to(tl.int64)
    N_HEAD_i64 = tl.cast(N_HEAD, tl.int64)
    T_LEN_i64 = tl.cast(T_LEN, tl.int64)
    H_SIZE_i64 = tl.cast(H_SIZE, tl.int64)
    pid_h_i64 = tl.cast(pid_h, tl.int64)
    CHUNK_LEN_i64 = tl.cast(CHUNK_LEN, tl.int64)

    ptr_mask = b_mask[:, None]
    state_mask = b_mask[:, None, None]

    base_ptr_off = (
        (b_range_i64[:, None] * N_HEAD_i64 * T_LEN_i64 * H_SIZE_i64)
        + (pid_h_i64 * T_LEN_i64 * H_SIZE_i64)
        + cols[None, :]
    )
    h0_base = (b_range_i64[:, None, None] * N_HEAD_i64 * H_SIZE_i64 * H_SIZE_i64) + (
        pid_h_i64 * H_SIZE_i64 * H_SIZE_i64
    )

    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]
    state = tl.load(
        H0 + h0_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)

    chunk_num_i64 = T_LEN_i64 // CHUNK_LEN_i64
    tau_base_batch = (
        b_range_i64 * N_HEAD_i64 * chunk_num_i64 + pid_h_i64 * chunk_num_i64
    )

    for t in range(0, T_LEN):
        t_off = t * H_SIZE

        rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )

        w_decay = tl.exp(-tl.exp(wv))
        sa_vec = tl.sum(state * av[:, None, :], axis=2)
        tl.store(
            SA_OUT + base_ptr_off + t_off,
            sa_vec.to(SA_OUT.dtype.element_ty),
            mask=ptr_mask,
        )

        state = (
            state * w_decay[:, None, :]
            + sa_vec[:, :, None] * bv[:, None, :]
            + vv[:, :, None] * kv[:, None, :]
        )

        y_vec = tl.sum(state * rv[:, None, :], axis=2)
        tl.store(
            OUT + base_ptr_off + t_off, y_vec.to(OUT.dtype.element_ty), mask=ptr_mask
        )

        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (tl.cast(t, tl.int64) + 1) // CHUNK_LEN_i64 - 1
            chkp_base = (
                (
                    b_range_i64[:, None, None]
                    * N_HEAD_i64
                    * chunk_num_i64
                    * H_SIZE_i64
                    * H_SIZE_i64
                )
                + (pid_h_i64 * chunk_num_i64 * H_SIZE_i64 * H_SIZE_i64)
                + (chkp_t * H_SIZE_i64 * H_SIZE_i64)
            )
            tl.store(
                STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                state,
                mask=state_mask,
            )

            # chunk 边界无条件执行 State Anomaly Neutralization。
            tau_v = tl.load(TAU + tau_base_batch + chkp_t, mask=b_mask, other=1.0).to(
                tl.float32
            )
            tau_safe = tl.maximum(tau_v, 1e-6)
            x2 = 2.0 * state / tau_safe[:, None, None]
            tnh = tl.where(
                state >= 0.0,
                1.0 - 2.0 / (tl.exp(x2) + 1.0),
                2.0 / (tl.exp(-x2) + 1.0) - 1.0,
            )
            state = tau_safe[:, None, None] * tnh


#  无 mask 反向
@triton.autotune(
    configs=[
        triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def rwkv7_sane_bwd_kernel(
    R,
    W,
    K,
    V,
    A,
    B_param,
    SA,
    STATE_CHKP,
    TAU,
    B_BATCH,
    N_HEAD,
    T_LEN,
    DY,
    DHT,
    DR,
    DW,
    DK,
    DV,
    DA,
    DB,
    DTAU,
    DH0,
    H_SIZE: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    MINI_BSZ: tl.constexpr,
):
    """RWKV-7-SANE 训练反向 Triton kernel（无 mask）。

    每个 block 处理一个 (batch, head)，逆序扫描 T 步，在每个 chunk 边界先算 dtau，
    再对下游梯度乘 sech2。

    Args:
        R, W, K, V, A, B_param: [B, H, T, K], bfloat16, row-major。前向输入。
        SA: [B, H, T, K], float32, row-major。前向保存的 sa。
        STATE_CHKP: [B, H, T//16, K, K], float32, row-major。SANE 之前的 state。
        TAU: [B, H, T//16], float32, row-major。
        DY: [B, H, T, K], bfloat16, row-major。输出梯度。
        DHT: [B, H, K, K], float32, row-major。最终 state 梯度。
        DR/DW/DK/DV/DA/DB: [B, H, T, K], bfloat16, row-major。输入梯度输出。
        DTAU: [B, H, T//16], float32, row-major。
        DH0: [B, H, K, K], float32, row-major。

    编译期宏同 rwkv7_sane_fwd_kernel。
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
    b_mask = b_range < B_BATCH
    cols = tl.arange(0, H_SIZE)
    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]

    b_range_i64 = b_range.to(tl.int64)
    N_HEAD_i64 = tl.cast(N_HEAD, tl.int64)
    T_LEN_i64 = tl.cast(T_LEN, tl.int64)
    H_SIZE_i64 = tl.cast(H_SIZE, tl.int64)
    pid_h_i64 = tl.cast(pid_h, tl.int64)
    CHUNK_LEN_i64 = tl.cast(CHUNK_LEN, tl.int64)

    ptr_mask = b_mask[:, None]
    state_mask = b_mask[:, None, None]

    base_ptr_off = (
        (b_range_i64[:, None] * N_HEAD_i64 * T_LEN_i64 * H_SIZE_i64)
        + (pid_h_i64 * T_LEN_i64 * H_SIZE_i64)
        + cols[None, :]
    )
    dht_base = (b_range_i64[:, None, None] * N_HEAD_i64 * H_SIZE_i64 * H_SIZE_i64) + (
        pid_h_i64 * H_SIZE_i64 * H_SIZE_i64
    )

    dS = tl.load(
        DHT + dht_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)
    S_t = tl.zeros([MINI_BSZ, H_SIZE, H_SIZE], dtype=tl.float32)

    chunk_num_i64 = T_LEN_i64 // CHUNK_LEN_i64
    tau_base_batch = (
        b_range_i64 * N_HEAD_i64 * chunk_num_i64 + pid_h_i64 * chunk_num_i64
    )

    for t in range(T_LEN - 1, -1, -1):
        t_off = t * H_SIZE

        rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )
        dyv = tl.load(DY + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )
        sav = tl.load(SA + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )

        w_decay = tl.exp(-tl.exp(wv))
        w_grad_factor = w_decay * (-tl.exp(wv))

        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (tl.cast(t, tl.int64) + 1) // CHUNK_LEN_i64 - 1
            chkp_base = (
                (
                    b_range_i64[:, None, None]
                    * N_HEAD_i64
                    * chunk_num_i64
                    * H_SIZE_i64
                    * H_SIZE_i64
                )
                + (pid_h_i64 * chunk_num_i64 * H_SIZE_i64 * H_SIZE_i64)
                + (chkp_t * H_SIZE_i64 * H_SIZE_i64)
            )
            S_t = tl.load(
                STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                mask=state_mask,
                other=0.0,
            ).to(tl.float32)

            # SANE 反向：先对下游梯度 dS 应用 SANE 导数。
            tau_v = tl.load(TAU + tau_base_batch + chkp_t, mask=b_mask, other=1.0).to(
                tl.float32
            )
            tau_safe = tl.maximum(tau_v, 1e-6)
            u = S_t / tau_safe[:, None, None]
            x2 = 2.0 * u
            tnh = tl.where(
                u >= 0.0,
                1.0 - 2.0 / (tl.exp(x2) + 1.0),
                2.0 / (tl.exp(-x2) + 1.0) - 1.0,
            )
            sech2 = 1.0 - tnh * tnh

            dtau_local = tl.sum(tl.sum(dS * (tnh - u * sech2), axis=2), axis=1)
            tl.store(DTAU + tau_base_batch + chkp_t, dtau_local, mask=b_mask)
            dS = dS * sech2

        dr = tl.sum(S_t * dyv[:, :, None], axis=1)
        tl.store(DR + base_ptr_off + t_off, dr.to(DR.dtype.element_ty), mask=ptr_mask)

        inv_w = 1.0 / (w_decay + 1e-6)
        S_t = (
            S_t - vv[:, :, None] * kv[:, None, :] - sav[:, :, None] * bv[:, None, :]
        ) * inv_w[:, None, :]

        dS = dS + dyv[:, :, None] * rv[:, None, :]

        dw = tl.sum(dS * S_t, axis=1) * w_grad_factor
        dk = tl.sum(dS * vv[:, :, None], axis=1)
        dv = tl.sum(dS * kv[:, None, :], axis=2)
        db = tl.sum(dS * sav[:, :, None], axis=1)
        dsa = tl.sum(dS * bv[:, None, :], axis=2)
        da = tl.sum(S_t * dsa[:, :, None], axis=1)

        tl.store(DW + base_ptr_off + t_off, dw.to(DW.dtype.element_ty), mask=ptr_mask)
        tl.store(DK + base_ptr_off + t_off, dk.to(DK.dtype.element_ty), mask=ptr_mask)
        tl.store(DV + base_ptr_off + t_off, dv.to(DV.dtype.element_ty), mask=ptr_mask)
        tl.store(DB + base_ptr_off + t_off, db.to(DB.dtype.element_ty), mask=ptr_mask)
        tl.store(DA + base_ptr_off + t_off, da.to(DA.dtype.element_ty), mask=ptr_mask)

        dS = dS * w_decay[:, None, :] + dsa[:, :, None] * av[:, None, :]

        if t == 0:
            tl.store(
                DH0 + dht_base + row_idx * H_SIZE + col_idx,
                dS.to(DH0.dtype.element_ty),
                mask=state_mask,
            )


#  带 mask 前向（chunk 边界按 mask 选择 SANE）
@triton.autotune(
    configs=[
        triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def rwkv7_sane_fwd_kernel_with_mask(
    R,
    W,
    K,
    V,
    A,
    B_param,
    TAU,
    MASK,
    H0,
    B_BATCH,
    N_HEAD,
    T_LEN,
    OUT,
    SA_OUT,
    STATE_CHKP,
    H_SIZE: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    MINI_BSZ: tl.constexpr,
):
    """RWKV-7-SANE 训练前向 Triton kernel（带 mask）。

    与无 mask 版本相同，但在 chunk 边界按 mask 选择是否执行 SANE。

    Args:
        R, W, K, V, A, B_param: [B, H, T, K], bfloat16, row-major。
        TAU: [B, H, T//16], float32, row-major。
        MASK: [B, T//16], float32, row-major。>0 执行 SANE。
        H0, OUT, SA_OUT, STATE_CHKP: 同无 mask 版本。
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
    b_mask = b_range < B_BATCH
    cols = tl.arange(0, H_SIZE)

    b_range_i64 = b_range.to(tl.int64)
    N_HEAD_i64 = tl.cast(N_HEAD, tl.int64)
    T_LEN_i64 = tl.cast(T_LEN, tl.int64)
    H_SIZE_i64 = tl.cast(H_SIZE, tl.int64)
    pid_h_i64 = tl.cast(pid_h, tl.int64)
    CHUNK_LEN_i64 = tl.cast(CHUNK_LEN, tl.int64)

    ptr_mask = b_mask[:, None]
    state_mask = b_mask[:, None, None]

    base_ptr_off = (
        (b_range_i64[:, None] * N_HEAD_i64 * T_LEN_i64 * H_SIZE_i64)
        + (pid_h_i64 * T_LEN_i64 * H_SIZE_i64)
        + cols[None, :]
    )
    h0_base = (b_range_i64[:, None, None] * N_HEAD_i64 * H_SIZE_i64 * H_SIZE_i64) + (
        pid_h_i64 * H_SIZE_i64 * H_SIZE_i64
    )
    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]

    state = tl.load(
        H0 + h0_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)

    chunk_num_i64 = T_LEN_i64 // CHUNK_LEN_i64
    tau_base_batch = (
        b_range_i64 * N_HEAD_i64 * chunk_num_i64 + pid_h_i64 * chunk_num_i64
    )
    mask_base_batch = b_range_i64 * chunk_num_i64

    for t in range(0, T_LEN):
        t_off = t * H_SIZE

        rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )

        w_decay = tl.exp(-tl.exp(wv))
        sa_vec = tl.sum(state * av[:, None, :], axis=2)
        tl.store(
            SA_OUT + base_ptr_off + t_off,
            sa_vec.to(SA_OUT.dtype.element_ty),
            mask=ptr_mask,
        )

        state = (
            state * w_decay[:, None, :]
            + sa_vec[:, :, None] * bv[:, None, :]
            + vv[:, :, None] * kv[:, None, :]
        )

        y_vec = tl.sum(state * rv[:, None, :], axis=2)
        tl.store(
            OUT + base_ptr_off + t_off, y_vec.to(OUT.dtype.element_ty), mask=ptr_mask
        )

        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (tl.cast(t, tl.int64) + 1) // CHUNK_LEN_i64 - 1
            chkp_base = (
                (
                    b_range_i64[:, None, None]
                    * N_HEAD_i64
                    * chunk_num_i64
                    * H_SIZE_i64
                    * H_SIZE_i64
                )
                + (pid_h_i64 * chunk_num_i64 * H_SIZE_i64 * H_SIZE_i64)
                + (chkp_t * H_SIZE_i64 * H_SIZE_i64)
            )
            tl.store(
                STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                state,
                mask=state_mask,
            )

            # 按 chunk-level mask 选择是否执行 State Anomaly Neutralization（blend 形式避免 warp 分支）。
            tau_v = tl.load(TAU + tau_base_batch + chkp_t, mask=b_mask, other=1.0).to(
                tl.float32
            )
            tau_safe = tl.maximum(tau_v, 1e-6)
            m_val = tl.load(MASK + mask_base_batch + chkp_t, mask=b_mask, other=0.0).to(
                tl.float32
            )
            x2 = 2.0 * state / tau_safe[:, None, None]
            sane_state = tau_safe[:, None, None] * tl.where(
                state >= 0.0,
                1.0 - 2.0 / (tl.exp(x2) + 1.0),
                2.0 / (tl.exp(-x2) + 1.0) - 1.0,
            )
            state = (
                state * (1.0 - m_val[:, None, None]) + sane_state * m_val[:, None, None]
            )


#  带 mask 反向
@triton.autotune(
    configs=[
        triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H_SIZE"],
)
@triton.jit
def rwkv7_sane_bwd_kernel_with_mask(
    R,
    W,
    K,
    V,
    A,
    B_param,
    SA,
    STATE_CHKP,
    TAU,
    MASK,
    B_BATCH,
    N_HEAD,
    T_LEN,
    DY,
    DHT,
    DR,
    DW,
    DK,
    DV,
    DA,
    DB,
    DTAU,
    DH0,
    H_SIZE: tl.constexpr,
    CHUNK_LEN: tl.constexpr,
    MINI_BSZ: tl.constexpr,
):
    """RWKV-7-SANE 训练反向 Triton kernel（带 mask）。

    与无 mask 版本相同，但在 chunk 边界按 mask 控制 dtau 与梯度 blend。

    Args:
        MASK: [B, T//16], float32, row-major。
        其余同 rwkv7_sane_bwd_kernel。
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
    b_mask = b_range < B_BATCH
    cols = tl.arange(0, H_SIZE)
    row_idx = tl.arange(0, H_SIZE)[None, :, None]
    col_idx = tl.arange(0, H_SIZE)[None, None, :]

    b_range_i64 = b_range.to(tl.int64)
    N_HEAD_i64 = tl.cast(N_HEAD, tl.int64)
    T_LEN_i64 = tl.cast(T_LEN, tl.int64)
    H_SIZE_i64 = tl.cast(H_SIZE, tl.int64)
    pid_h_i64 = tl.cast(pid_h, tl.int64)
    CHUNK_LEN_i64 = tl.cast(CHUNK_LEN, tl.int64)

    ptr_mask = b_mask[:, None]
    state_mask = b_mask[:, None, None]

    base_ptr_off = (
        (b_range_i64[:, None] * N_HEAD_i64 * T_LEN_i64 * H_SIZE_i64)
        + (pid_h_i64 * T_LEN_i64 * H_SIZE_i64)
        + cols[None, :]
    )
    dht_base = (b_range_i64[:, None, None] * N_HEAD_i64 * H_SIZE_i64 * H_SIZE_i64) + (
        pid_h_i64 * H_SIZE_i64 * H_SIZE_i64
    )

    dS = tl.load(
        DHT + dht_base + row_idx * H_SIZE + col_idx, mask=state_mask, other=0.0
    ).to(tl.float32)
    S_t = tl.zeros([MINI_BSZ, H_SIZE, H_SIZE], dtype=tl.float32)

    chunk_num_i64 = T_LEN_i64 // CHUNK_LEN_i64
    tau_base_batch = (
        b_range_i64 * N_HEAD_i64 * chunk_num_i64 + pid_h_i64 * chunk_num_i64
    )
    mask_base_batch = b_range_i64 * chunk_num_i64

    for t in range(T_LEN - 1, -1, -1):
        t_off = t * H_SIZE

        rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
        bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )
        dyv = tl.load(DY + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )
        sav = tl.load(SA + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(
            tl.float32
        )

        w_decay = tl.exp(-tl.exp(wv))
        w_grad_factor = w_decay * (-tl.exp(wv))

        if (t + 1) % CHUNK_LEN == 0:
            chkp_t = (tl.cast(t, tl.int64) + 1) // CHUNK_LEN_i64 - 1
            chkp_base = (
                (
                    b_range_i64[:, None, None]
                    * N_HEAD_i64
                    * chunk_num_i64
                    * H_SIZE_i64
                    * H_SIZE_i64
                )
                + (pid_h_i64 * chunk_num_i64 * H_SIZE_i64 * H_SIZE_i64)
                + (chkp_t * H_SIZE_i64 * H_SIZE_i64)
            )
            S_t = tl.load(
                STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                mask=state_mask,
                other=0.0,
            ).to(tl.float32)

            tau_v = tl.load(TAU + tau_base_batch + chkp_t, mask=b_mask, other=1.0).to(
                tl.float32
            )
            tau_safe = tl.maximum(tau_v, 1e-6)
            m_val = tl.load(MASK + mask_base_batch + chkp_t, mask=b_mask, other=0.0).to(
                tl.float32
            )

            u = S_t / tau_safe[:, None, None]
            x2 = 2.0 * u
            tnh = tl.where(
                u >= 0.0,
                1.0 - 2.0 / (tl.exp(x2) + 1.0),
                2.0 / (tl.exp(-x2) + 1.0) - 1.0,
            )
            sech2 = 1.0 - tnh * tnh
            blend = (1.0 - m_val) + m_val * sech2

            dtau_local = tl.sum(
                tl.sum(dS * m_val[:, None, None] * (tnh - u * sech2), axis=2),
                axis=1,
            )
            tl.store(DTAU + tau_base_batch + chkp_t, dtau_local, mask=b_mask)
            dS = dS * blend

        dr = tl.sum(S_t * dyv[:, :, None], axis=1)
        tl.store(DR + base_ptr_off + t_off, dr.to(DR.dtype.element_ty), mask=ptr_mask)

        inv_w = 1.0 / (w_decay + 1e-6)
        S_t = (
            S_t - vv[:, :, None] * kv[:, None, :] - sav[:, :, None] * bv[:, None, :]
        ) * inv_w[:, None, :]

        dS = dS + dyv[:, :, None] * rv[:, None, :]

        dw = tl.sum(dS * S_t, axis=1) * w_grad_factor
        dk = tl.sum(dS * vv[:, :, None], axis=1)
        dv = tl.sum(dS * kv[:, None, :], axis=2)
        db = tl.sum(dS * sav[:, :, None], axis=1)
        dsa = tl.sum(dS * bv[:, None, :], axis=2)
        da = tl.sum(S_t * dsa[:, :, None], axis=1)

        tl.store(DW + base_ptr_off + t_off, dw.to(DW.dtype.element_ty), mask=ptr_mask)
        tl.store(DK + base_ptr_off + t_off, dk.to(DK.dtype.element_ty), mask=ptr_mask)
        tl.store(DV + base_ptr_off + t_off, dv.to(DV.dtype.element_ty), mask=ptr_mask)
        tl.store(DB + base_ptr_off + t_off, db.to(DB.dtype.element_ty), mask=ptr_mask)
        tl.store(DA + base_ptr_off + t_off, da.to(DA.dtype.element_ty), mask=ptr_mask)

        dS = dS * w_decay[:, None, :] + dsa[:, :, None] * av[:, None, :]

        if t == 0:
            tl.store(
                DH0 + dht_base + row_idx * H_SIZE + col_idx,
                dS.to(DH0.dtype.element_ty),
                mask=state_mask,
            )
