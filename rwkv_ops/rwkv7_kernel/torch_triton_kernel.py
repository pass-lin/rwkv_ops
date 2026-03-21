import torch
from .triton_kernel import *


class TritonWindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, a, b, h0):
        # 确保输入是标准的 [B, N_HEAD, T_LEN, H_SIZE] 且连续
        # w, q, k, v, a, b 已经在 generalized_delta_rule 中处理过连续性
        B, N, T, H = w.shape
        DTYPE = q.dtype
        CHUNK_LEN = 16

        if T % CHUNK_LEN != 0:
            raise ValueError(f"Sequence length T={T} must be divisible by {CHUNK_LEN}")
        if H != 64:
            raise ValueError(
                f"Triton kernel currently only supports Head Size = 64, got {H}"
            )

        # 1. 分配输出与中间变量
        # OUT 使用输入相同类型 (通常 bf16), SA 和 CHKP 强制使用 fp32 保证梯度精度
        out = torch.empty_like(v)
        sa_out = torch.empty(B, N, T, H, dtype=torch.float32, device=w.device)
        state_chkp = torch.empty(
            B, N, T // CHUNK_LEN, H, H, dtype=torch.float32, device=w.device
        )

        # 定义 Grid 分块策略
        def grid(META):
            return ((B + META["MINI_BSZ"] - 1) // META["MINI_BSZ"], N)

        # 2. 启动前向内核
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

        # 3. 保存反向传播所需的张量
        ctx.save_for_backward(w, q, k, v, a, b, state_chkp, sa_out)

        # 4. 获取并返回最终状态
        # state_chkp 最后一位就是最终时刻的状态
        last_state = state_chkp[:, :, -1, :, :].clone()

        return out.to(DTYPE), last_state

    @staticmethod
    def backward(ctx, dy, dht):
        # dy: [B, N, T, H] (对应输出 y 的梯度)
        # dht: [B, N, H, H] (对应最终状态的梯度)

        # 1. 恢复前向变量
        w, q, k, v, a, b, state_chkp, sa_out = ctx.saved_tensors
        B, N, T, H = w.shape
        DTYPE = q.dtype
        CHUNK_LEN = 16

        # 2. 准备梯度输出张量
        dw = torch.empty_like(w)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        # dh0 始终使用 float32
        dh0 = torch.empty(B, N, H, H, dtype=torch.float32, device=w.device)

        # 3. 预处理输入梯度
        dy = dy.contiguous()
        if dht is None:
            dht = torch.zeros(B, N, H, H, dtype=torch.float32, device=w.device)
        else:
            dht = dht.contiguous().to(torch.float32)

        # 定义 Grid
        def grid(META):
            return ((B + META["MINI_BSZ"] - 1) // META["MINI_BSZ"], N)

        # 4. 启动反向内核
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

        # 5. 返回各输入的梯度 (必须与 forward 的输入顺序一致)
        # 返回顺序: w, q, k, v, a, b, h0
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
        # 确保输入是标准的连续张量 [B, N_HEAD, T_LEN, H_SIZE]
        B, N, T, H = w.shape
        DTYPE = q.dtype
        CHUNK_LEN = 16

        if T % CHUNK_LEN != 0:
            raise ValueError(f"Sequence length T={T} must be divisible by {CHUNK_LEN}")
        if H != 64:
            raise ValueError(
                f"Triton kernel currently only supports Head Size = 64, got {H}"
            )

        # mask 必须是连续的张量，通常传入的形状可能是 [B, T] 或 [B, T, 1, 1]
        # Triton kernel 中我们按照 [B, T] 连续内存读取，因此如果维度多余需要 squeeze，然后 contiguous
        if mask.dim() > 2:
            mask = mask.view(B, T)
        mask = mask.contiguous().to(torch.float32)  # Triton内核中统一转为 fp32 计算

        # 分配输出与中间变量
        out = torch.empty_like(v)
        sa_out = torch.empty(B, N, T, H, dtype=torch.float32, device=w.device)
        state_chkp = torch.empty(
            B, N, T // CHUNK_LEN, H, H, dtype=torch.float32, device=w.device
        )

        # 定义 Grid 分块策略
        def grid(META):
            return ((B + META["MINI_BSZ"] - 1) // META["MINI_BSZ"], N)

        # 启动带 Mask 的前向内核
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

        # 保存反向传播所需的张量 (注意要保存 mask)
        ctx.save_for_backward(w, q, k, v, a, b, mask, state_chkp, sa_out)

        # 获取并返回最终状态
        last_state = state_chkp[:, :, -1, :, :].clone()

        return out.to(DTYPE), last_state

    @staticmethod
    def backward(ctx, dy, dht):
        # 1. 恢复前向变量
        w, q, k, v, a, b, mask, state_chkp, sa_out = ctx.saved_tensors
        B, N, T, H = w.shape
        DTYPE = q.dtype
        CHUNK_LEN = 16

        # 2. 准备梯度输出张量
        dw = torch.empty_like(w)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        dh0 = torch.empty(B, N, H, H, dtype=torch.float32, device=w.device)

        # 3. 预处理输入梯度
        dy = dy.contiguous()
        if dht is None:
            dht = torch.zeros(B, N, H, H, dtype=torch.float32, device=w.device)
        else:
            dht = dht.contiguous().to(torch.float32)

        # 定义 Grid
        def grid(META):
            return ((B + META["MINI_BSZ"] - 1) // META["MINI_BSZ"], N)

        # 4. 启动带 Mask 的反向内核
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

        # 5. 返回梯度
        # 注意：forward 的输入签名是 (ctx, w, q, k, v, a, b, mask, h0)
        # 所以我们需要返回 8 个梯度：w, q, k, v, a, b, mask, h0
        # mask 不需要梯度，因此返回 None
        return (
            dw.to(DTYPE),
            dq.to(DTYPE),
            dk.to(DTYPE),
            dv.to(DTYPE),
            da.to(DTYPE),
            db.to(DTYPE),
            None,  # mask 的梯度为 None
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
    """
    统一接口函数。
    支持输入形状:
        head_first=False: [B, T, N, H] (Keras 默认)
        head_first=True : [B, N, T, H] (Triton 算子标准)
    """
    if w.device.type != "cuda":
        raise RuntimeError("Triton kernel only supports CUDA devices.")

    # 1. 统一转换为 [B, N, T, H] 的 Head-First 连续张量
    if not head_first:
        r = r.transpose(1, 2)
        w = w.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        a = a.transpose(1, 2)
        b = b.transpose(1, 2)
        # mask 处理 (通常为 [B, T] 或 [B, T, 1, 1])
        # 这里预留 Mask 转置逻辑，如果 mask 是 [B, T]，则不需要 transpose(1, 2)

    # 确保 Contiguous 极度重要，Triton 内核依赖于此进行偏移计算
    r, w, k, v, a, b = [x.contiguous() for x in [r, w, k, v, a, b]]

    B, N, T, H = w.shape

    # 2. 初始状态处理
    if initial_state is None:
        initial_state = torch.zeros(B, N, H, H, dtype=torch.float32, device=r.device)
    else:
        # 确保初始状态也是连续的 float32
        initial_state = initial_state.to(torch.float32).contiguous()

    # 3. 分发到 Autograd Function
    if mask is None:
        # 无 Mask 版本
        out, state = TritonWindBackstepping.apply(w, r, k, v, a, b, initial_state)
    else:
        if not mask.is_cuda:
            mask = mask.to(r.device)
        out, state = TritonWindBacksteppingWithMask.apply(
            w, r, k, v, a, b, mask, initial_state
        )

    # 4. 如果原始输入不是 head_first，则将输出转置回去 [B, N, T, H] -> [B, T, N, H]
    if not head_first:
        out = out.transpose(1, 2)

    return (out, state) if output_final_state else out
