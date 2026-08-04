"""PyTorch 版 RWKV7 Triton kernel 封装。"""

import torch
from .triton_kernel import (
    rwkv7_bwd_kernel,
    rwkv7_bwd_kernel_with_mask,
    rwkv7_fwd_kernel,
    rwkv7_fwd_kernel_with_mask,
)


class TritonWindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, a, b, h0):
        # 输入已保证为 [B, N_HEAD, T_LEN, H_SIZE] 且连续。
        B, N, T, H = w.shape
        DTYPE = q.dtype
        CHUNK_LEN = 16

        if T % CHUNK_LEN != 0:
            raise ValueError(f"Sequence length T={T} must be divisible by {CHUNK_LEN}")
        if H != 64:
            raise ValueError(
                f"Triton kernel currently only supports Head Size = 64, got {H}"
            )

        # OUT 复用输入 dtype，SA 与 checkpoint 固定 fp32 以保证梯度精度。
        out = torch.empty_like(v)
        sa_out = torch.empty(B, N, T, H, dtype=torch.float32, device=w.device)
        state_chkp = torch.empty(
            B, N, T // CHUNK_LEN, H, H, dtype=torch.float32, device=w.device
        )

        def grid(META):
            return ((B + META["MINI_BSZ"] - 1) // META["MINI_BSZ"], N)

        rwkv7_fwd_kernel[grid](
            R=q,
            W=w,
            K=k,
            V=v,
            A=a,
            B_param=b,
            H0=h0,
            OUT=out,
            SA_OUT=sa_out,
            STATE_CHKP=state_chkp,
            N_HEAD=N,
            B_BATCH=B,
            T_LEN=T,
            H_SIZE=H,
            CHUNK_LEN=CHUNK_LEN,
        )

        ctx.save_for_backward(w, q, k, v, a, b, state_chkp, sa_out)

        # state_chkp 最后一个 chunk 即最终时刻状态。
        last_state = state_chkp[:, :, -1, :, :].clone()

        return out.to(DTYPE), last_state

    @staticmethod
    def backward(ctx, dy, dht):
        # dy: [B, N, T, H]，dht: [B, N, H, H]。
        w, q, k, v, a, b, state_chkp, sa_out = ctx.saved_tensors
        B, N, T, H = w.shape
        DTYPE = q.dtype
        CHUNK_LEN = 16

        dw = torch.empty_like(w)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        dh0 = torch.empty(B, N, H, H, dtype=torch.float32, device=w.device)

        dy = dy.contiguous()
        if dht is None:
            dht = torch.zeros(B, N, H, H, dtype=torch.float32, device=w.device)
        else:
            dht = dht.contiguous().to(torch.float32)

        def grid(META):
            return ((B + META["MINI_BSZ"] - 1) // META["MINI_BSZ"], N)

        rwkv7_bwd_kernel[grid](
            R=q,
            W=w,
            K=k,
            V=v,
            A=a,
            B_param=b,
            SA=sa_out,
            STATE_CHKP=state_chkp,
            DY=dy,
            DHT=dht,
            B_BATCH=B,
            N_HEAD=N,
            T_LEN=T,
            DR=dq,
            DW=dw,
            DK=dk,
            DV=dv,
            DA=da,
            DB=db,
            DH0=dh0,
            H_SIZE=H,
            CHUNK_LEN=CHUNK_LEN,
        )

        # 返回顺序必须与 forward 输入顺序一致：w, q, k, v, a, b, h0。
        return (
            dw.to(DTYPE),
            dq.to(DTYPE),
            dk.to(DTYPE),
            dv.to(DTYPE),
            da.to(DTYPE),
            db.to(DTYPE),
            dh0,
        )


class TritonWindBacksteppingWithMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, a, b, mask, h0):
        # 输入已保证为 [B, N_HEAD, T_LEN, H_SIZE] 且连续。
        B, N, T, H = w.shape
        DTYPE = q.dtype
        CHUNK_LEN = 16

        if T % CHUNK_LEN != 0:
            raise ValueError(f"Sequence length T={T} must be divisible by {CHUNK_LEN}")
        if H != 64:
            raise ValueError(
                f"Triton kernel currently only supports Head Size = 64, got {H}"
            )

        # Triton kernel 按 [B, T] 连续内存读取 mask，多余维度需先 squeeze。
        if mask.dim() > 2:
            mask = mask.view(B, T)
        mask = mask.contiguous().to(torch.float32)

        out = torch.empty_like(v)
        sa_out = torch.empty(B, N, T, H, dtype=torch.float32, device=w.device)
        state_chkp = torch.empty(
            B, N, T // CHUNK_LEN, H, H, dtype=torch.float32, device=w.device
        )

        def grid(META):
            return ((B + META["MINI_BSZ"] - 1) // META["MINI_BSZ"], N)

        rwkv7_fwd_kernel_with_mask[grid](
            R=q,
            W=w,
            K=k,
            V=v,
            A=a,
            B_param=b,
            MASK=mask,
            H0=h0,
            B_BATCH=B,
            N_HEAD=N,
            T_LEN=T,
            OUT=out,
            SA_OUT=sa_out,
            STATE_CHKP=state_chkp,
            H_SIZE=H,
            CHUNK_LEN=CHUNK_LEN,
        )

        ctx.save_for_backward(w, q, k, v, a, b, mask, state_chkp, sa_out)

        last_state = state_chkp[:, :, -1, :, :].clone()

        return out.to(DTYPE), last_state

    @staticmethod
    def backward(ctx, dy, dht):
        w, q, k, v, a, b, mask, state_chkp, sa_out = ctx.saved_tensors
        B, N, T, H = w.shape
        DTYPE = q.dtype
        CHUNK_LEN = 16

        dw = torch.empty_like(w)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        dh0 = torch.empty(B, N, H, H, dtype=torch.float32, device=w.device)

        dy = dy.contiguous()
        if dht is None:
            dht = torch.zeros(B, N, H, H, dtype=torch.float32, device=w.device)
        else:
            dht = dht.contiguous().to(torch.float32)

        def grid(META):
            return ((B + META["MINI_BSZ"] - 1) // META["MINI_BSZ"], N)

        rwkv7_bwd_kernel_with_mask[grid](
            R=q,
            W=w,
            K=k,
            V=v,
            A=a,
            B_param=b,
            MASK=mask,
            SA=sa_out,
            STATE_CHKP=state_chkp,
            B_BATCH=B,
            N_HEAD=N,
            T_LEN=T,
            DY=dy,
            DHT=dht,
            DR=dq,
            DW=dw,
            DK=dk,
            DV=dv,
            DA=da,
            DB=db,
            DH0=dh0,
            H_SIZE=H,
            CHUNK_LEN=CHUNK_LEN,
        )

        # forward 输入顺序为 (ctx, w, q, k, v, a, b, mask, h0)；mask 无梯度。
        return (
            dw.to(DTYPE),
            dq.to(DTYPE),
            dk.to(DTYPE),
            dv.to(DTYPE),
            da.to(DTYPE),
            db.to(DTYPE),
            None,
            dh0,
        )


def generalized_delta_rule(
    r,
    w,
    k,
    v,
    a,
    b,
    initial_state=None,
    output_final_state: bool = True,
    head_first: bool = False,
    mask=None,
):
    """RWKV-7 chunkwise 训练算子（PyTorch Triton 实现）。

    Args:
        r, w, k, v, a, b: [B, T, H, K]，bfloat16。T 必须被 16 整除，H 必须为 64。
        initial_state: [B, H, K, K]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, K]）。
        mask: [B, T] 或 [B, T, 1, 1]，float32，1 表示更新状态、0 表示冻结状态。

    Returns:
        out: [B, T, H, K]，与输入同 dtype。
        final_state: [B, H, K, K]，float32；仅当 output_final_state=True 时返回。

    Raises:
        RuntimeError: 输入不在 CUDA 设备上。
        ValueError: T 不被 16 整除，或 H 不等于 64。
    """
    if w.device.type != "cuda":
        raise RuntimeError("Triton kernel only supports CUDA devices.")

    if not head_first:
        r = r.transpose(1, 2)
        w = w.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        a = a.transpose(1, 2)
        b = b.transpose(1, 2)

    # Triton 内核依赖连续内存做指针偏移。
    r, w, k, v, a, b = [x.contiguous() for x in [r, w, k, v, a, b]]

    B, N, T, H = w.shape

    if initial_state is None:
        initial_state = torch.zeros(B, N, H, H, dtype=torch.float32, device=r.device)
    else:
        initial_state = initial_state.to(torch.float32).contiguous()

    if mask is None:
        out, state = TritonWindBackstepping.apply(w, r, k, v, a, b, initial_state)
    else:
        if not mask.is_cuda:
            mask = mask.to(r.device)
        out, state = TritonWindBacksteppingWithMask.apply(
            w, r, k, v, a, b, mask, initial_state
        )

    if not head_first:
        out = out.transpose(1, 2)

    return (out, state) if output_final_state else out
