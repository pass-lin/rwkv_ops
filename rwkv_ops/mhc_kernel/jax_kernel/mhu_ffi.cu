#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <xla/ffi/api/ffi.h>
#include <vector>
#include <cstdint>

// 公共头文件路径
#include "../common_kernel/include/mhc_types.h"
#include "../common_kernel/kernels/sinkhorn_knopp.cuh"
#include "../common_kernel/kernels/rmsnorm.cuh"
#include "../common_kernel/kernels/stream_mix.cuh"
#include "../common_kernel/kernels/stream_aggregate.cuh"
#include "../common_kernel/kernels/stream_distribute.cuh"
#include "../common_kernel/kernels/mhc_post_op.cuh"
#include "../common_kernel/kernels/mhc_pre_op.cuh"
namespace ffi = xla::ffi;

/* -------------------- Sinkhorn Knopp FFI -------------------- */

// 前向FFI处理器
static ffi::Error SinkhornFwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> inp,        // 输入: [B, T, N, N]
    ffi::ResultBuffer<ffi::F32> out,  // 输出: [B, T, N, N]
    std::int32_t num_iters,           // 显式使用 std::int32_t
    float eps                         // float 本身就是32位
) {
    // 获取张量维度
    auto dims = inp.dimensions();
    int64_t B = dims[0];
    int64_t T = dims[1];
    int64_t N = dims[2];
    
    const float* inp_ptr = inp.typed_data();
    float* out_ptr = out->typed_data();
    
    // 批量调用sinkhorn前向
    for (int64_t b = 0; b < B * T; ++b) {
        mhc::sinkhorn_knopp_forward(
            out_ptr + b * N * N,
            inp_ptr + b * N * N,
            static_cast<int>(N),
            static_cast<int>(N),
            num_iters,  // 已经是int32
            eps,
            stream
        );
    }
    
    return ffi::Error::Success();
}

// 反向FFI处理器
static ffi::Error SinkhornBwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> grad,       // 梯度: [B, T, N, N]
    ffi::Buffer<ffi::F32> out_fwd,    // 前向输出: [B, T, N, N]
    ffi::Buffer<ffi::F32> inp,        // 原始输入: [B, T, N, N]
    ffi::ResultBuffer<ffi::F32> d_inp, // 输入梯度: [B, T, N, N]
    std::int32_t num_iters,           // 显式使用 std::int32_t
    float eps
) {
    // 获取张量维度
    auto dims = grad.dimensions();
    int64_t B = dims[0];
    int64_t T = dims[1];
    int64_t N = dims[2];
    
    const float* grad_ptr = grad.typed_data();
    const float* out_fwd_ptr = out_fwd.typed_data();
    const float* inp_ptr = inp.typed_data();
    float* d_inp_ptr = d_inp->typed_data();
    
    // 批量调用sinkhorn反向
    for (int64_t b = 0; b < B * T; ++b) {
        mhc::sinkhorn_knopp_backward(
            d_inp_ptr + b * N * N,
            grad_ptr + b * N * N,
            out_fwd_ptr + b * N * N,
            inp_ptr + b * N * N,
            static_cast<int>(N),
            num_iters,  // 已经是int32
            eps,
            stream
        );
    }
    
    return ffi::Error::Success();
}

/* -------------------- FFI 符号注册 -------------------- */

// 前向符号注册
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SinkhornFwd, SinkhornFwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // inp
        .Ret<ffi::Buffer<ffi::F32>>()      // out
        .Attr<std::int32_t>("num_iters")    // 显式指定32位整数
        .Attr<float>("eps")                 // float 默认是32位
);

// 反向符号注册
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SinkhornBwd, SinkhornBwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // grad
        .Arg<ffi::Buffer<ffi::F32>>()      // out_fwd
        .Arg<ffi::Buffer<ffi::F32>>()      // inp
        .Ret<ffi::Buffer<ffi::F32>>()      // d_inp
        .Attr<std::int32_t>("num_iters")    // 显式指定32位整数
        .Attr<float>("eps")                 // float 默认是32位
);

static ffi::Error RMSNormFwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> inp,        // 输入: [N, C]
    ffi::ResultBuffer<ffi::BF16> out,  // 输出: [N, C]
    float eps
) {
    auto dims = inp.dimensions();
    int64_t N = dims[0];
    int64_t C = dims[1];
    
    const nv_bfloat16* inp_ptr = reinterpret_cast<const nv_bfloat16*>(inp.typed_data());
    nv_bfloat16* out_ptr = reinterpret_cast<nv_bfloat16*>(out->typed_data());
    
    // 调用包装函数
    mhc::rmsnorm_forward(out_ptr, inp_ptr, N, C, eps, stream);
    
    return ffi::Error::Success();
}

// 反向FFI处理器
static ffi::Error RMSNormBwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> grad,       // 梯度: [N, C]
    ffi::Buffer<ffi::BF16> inp,        // 原始输入: [N, C]
    ffi::ResultBuffer<ffi::BF16> dx,   // 输入梯度: [N, C]
    float eps
) {
    auto dims = grad.dimensions();
    int64_t N = dims[0];
    int64_t C = dims[1];
    
    const nv_bfloat16* grad_ptr = reinterpret_cast<const nv_bfloat16*>(grad.typed_data());
    const nv_bfloat16* inp_ptr = reinterpret_cast<const nv_bfloat16*>(inp.typed_data());
    nv_bfloat16* dx_ptr = reinterpret_cast<nv_bfloat16*>(dx->typed_data());
    
    // 调用包装函数
    mhc::rmsnorm_backward(dx_ptr, grad_ptr, inp_ptr, N, C, eps, stream);
    
    return ffi::Error::Success();
}

/* -------------------- 注册 FFI 符号 -------------------- */

// 在文件末尾追加注册
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RMSNormFwd, RMSNormFwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()      // inp
        .Ret<ffi::Buffer<ffi::BF16>>()      // out
        .Attr<float>("eps")                  // eps
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RMSNormBwd, RMSNormBwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()      // grad
        .Arg<ffi::Buffer<ffi::BF16>>()      // inp
        .Ret<ffi::Buffer<ffi::BF16>>()      // dx
        .Attr<float>("eps")                  // eps
);

/* -------------------- Stream Mix FFI -------------------- */

// 前向FFI处理器
static ffi::Error StreamMixFwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> inp,        // 输入: [B, T, n, C]
    ffi::Buffer<ffi::F32> M,           // 权重: [B, T, n, n]
    ffi::ResultBuffer<ffi::BF16> out   // 输出: [B, T, n, C]
) {
    auto dims = inp.dimensions();
    int64_t B = dims[0];
    int64_t T = dims[1];
    int64_t n = dims[2];
    int64_t C = dims[3];
    
    const nv_bfloat16* inp_ptr = reinterpret_cast<const nv_bfloat16*>(inp.typed_data());
    const float* M_ptr = M.typed_data();
    nv_bfloat16* out_ptr = reinterpret_cast<nv_bfloat16*>(out->typed_data());
    
    // 调用包装函数
    mhc::stream_mix_forward(out_ptr, inp_ptr, M_ptr, B, T, static_cast<int>(n), C, stream);
    
    return ffi::Error::Success();
}

// 反向FFI处理器
// 修改1: 函数签名
static ffi::Error StreamMixBwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> grad,        // 从 BF16 改为 F32
    ffi::Buffer<ffi::BF16> inp,
    ffi::Buffer<ffi::F32> M,
    ffi::ResultBuffer<ffi::BF16> d_inp,
    ffi::ResultBuffer<ffi::F32> d_M
) {
    auto dims = grad.dimensions();  // 现在用 grad 获取维度
    int64_t B = dims[0];
    int64_t T = dims[1];
    int64_t n = dims[2];
    int64_t C = dims[3];
    
    const float* grad_ptr = grad.typed_data();  // 直接获取 float*
    const nv_bfloat16* inp_ptr = reinterpret_cast<const nv_bfloat16*>(inp.typed_data());
    const float* M_ptr = M.typed_data();
    nv_bfloat16* d_inp_ptr = reinterpret_cast<nv_bfloat16*>(d_inp->typed_data());
    float* d_M_ptr = d_M->typed_data();
    
    mhc::stream_mix_backward(d_inp_ptr, d_M_ptr, grad_ptr, inp_ptr, M_ptr, 
                            B, T, static_cast<int>(n), C, stream);
    
    return ffi::Error::Success();
}



/* -------------------- 注册 FFI 符号 -------------------- */

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    StreamMixFwd, StreamMixFwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()      // inp
        .Arg<ffi::Buffer<ffi::F32>>()      // M
        .Ret<ffi::Buffer<ffi::BF16>>()      // out
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    StreamMixBwd, StreamMixBwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // grad: F32
        .Arg<ffi::Buffer<ffi::BF16>>()      // inp: BF16
        .Arg<ffi::Buffer<ffi::F32>>()      // M: F32
        .Ret<ffi::Buffer<ffi::BF16>>()      // d_inp: BF16
        .Ret<ffi::Buffer<ffi::F32>>()      // d_M: F32
);
/* -------------------- Stream Aggregate FFI -------------------- */

// 前向FFI处理器
static ffi::Error StreamAggregateFwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> inp,        // 输入: [B, T, n, C]
    ffi::Buffer<ffi::F32> H_pre,       // 权重: [B, T, n] 或 [n]
    ffi::ResultBuffer<ffi::BF16> out,  // 输出: [B, T, C]
    bool per_token                     // 是否为per-token权重模式
) {
    auto dims = inp.dimensions();
    int64_t B = dims[0];
    int64_t T = dims[1];
    int64_t n = dims[2];
    int64_t C = dims[3];
    
    const nv_bfloat16* inp_ptr = reinterpret_cast<const nv_bfloat16*>(inp.typed_data());
    const float* H_pre_ptr = H_pre.typed_data();
    nv_bfloat16* out_ptr = reinterpret_cast<nv_bfloat16*>(out->typed_data());
    
    // 调用包装函数（注意：内部会自动处理per_token逻辑）
    mhc::stream_aggregate_forward(
        out_ptr, inp_ptr, H_pre_ptr, 
        B * T, static_cast<int>(n), C, per_token, stream
    );
    
    return ffi::Error::Success();
}

// 反向FFI处理器
static ffi::Error StreamAggregateBwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> grad,        // 梯度: [B, T, C] (float32)
    ffi::Buffer<ffi::BF16> inp,        // 原始输入: [B, T, n, C]
    ffi::Buffer<ffi::F32> H_pre,       // 权重: [B, T, n] 或 [n]
    ffi::ResultBuffer<ffi::BF16> d_inp,      // 输入梯度: [B, T, n, C]
    ffi::ResultBuffer<ffi::F32> d_H_pre,     // 权重梯度: [B, T, n] 或 [n]
    bool per_token                     // 是否为per-token权重模式
) {
    auto dims = inp.dimensions();
    int64_t B = dims[0];
    int64_t T = dims[1];
    int64_t n = dims[2];
    int64_t C = dims[3];
    
    const float* grad_ptr = grad.typed_data();
    const nv_bfloat16* inp_ptr = reinterpret_cast<const nv_bfloat16*>(inp.typed_data());
    const float* H_pre_ptr = H_pre.typed_data();
    nv_bfloat16* d_inp_ptr = reinterpret_cast<nv_bfloat16*>(d_inp->typed_data());
    float* d_H_pre_ptr = d_H_pre->typed_data();
    
    // 调用包装函数（内部会处理per_token逻辑和梯度累加）
    mhc::stream_aggregate_backward(
        d_inp_ptr, d_H_pre_ptr, grad_ptr, inp_ptr, H_pre_ptr, 
        B * T, static_cast<int>(n), C, per_token, stream
    );
    
    return ffi::Error::Success();
}

/* -------------------- 注册 FFI 符号 -------------------- */

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    StreamAggregateFwd, StreamAggregateFwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()      // inp
        .Arg<ffi::Buffer<ffi::F32>>()      // H_pre
        .Ret<ffi::Buffer<ffi::BF16>>()      // out
        .Attr<bool>("per_token")            // 权重模式
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    StreamAggregateBwd, StreamAggregateBwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // grad
        .Arg<ffi::Buffer<ffi::BF16>>()      // inp
        .Arg<ffi::Buffer<ffi::F32>>()      // H_pre
        .Ret<ffi::Buffer<ffi::BF16>>()      // d_inp
        .Ret<ffi::Buffer<ffi::F32>>()      // d_H_pre
        .Attr<bool>("per_token")            // 权重模式
);

/* -------------------- Stream Distribute FFI -------------------- */

// 前向：[B, T, C] (BF16), [B, T, n] (F32) -> [B, T, n, C] (BF16)
static ffi::Error StreamDistributeFwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> inp,     // [B, T, C]
    ffi::Buffer<ffi::F32> H_post,   // [B, T, n]
    ffi::ResultBuffer<ffi::BF16> out // [B, T, n, C]
) {
    auto dims_inp = inp.dimensions();
    auto dims_h = H_post.dimensions();
    
    int64_t B = dims_inp[0];
    int64_t T = dims_inp[1];
    int64_t C = dims_inp[2];
    int64_t n = dims_h[2];

    // blockIdx.x 覆盖 B*T*C，blockIdx.y 覆盖 n
    dim3 threads(256);
    dim3 blocks((B * T * C + 255) / 256, n);

    mhc::stream_distribute_fwd_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<mhc::floatX*>(out->typed_data()),
        reinterpret_cast<const mhc::floatX*>(inp.typed_data()),
        H_post.typed_data(),
        B, T, static_cast<int>(n), C
    );

    return ffi::Error::Success();
}

// 反向
static ffi::Error StreamDistributeBwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> grad,    // [B, T, n, C]
    ffi::Buffer<ffi::BF16> inp,     // [B, T, C]
    ffi::Buffer<ffi::F32> H_post,   // [B, T, n]
    ffi::ResultBuffer<ffi::BF16> d_inp,   // [B, T, C]
    ffi::ResultBuffer<ffi::F32> d_H_post  // [B, T, n]
) {
    auto dims = grad.dimensions();
    int64_t B = dims[0];
    int64_t T = dims[1];
    int64_t n = dims[2];
    int64_t C = dims[3];

    // 1. 计算 dx: [B, T, C]
    dim3 threads(256);
    dim3 blocks_dx((B * T * C + 255) / 256);
    mhc::stream_distribute_bwd_dx_kernel<<<blocks_dx, threads, 0, stream>>>(
        reinterpret_cast<mhc::floatX*>(d_inp->typed_data()),
        reinterpret_cast<const mhc::floatX*>(grad.typed_data()),
        H_post.typed_data(),
        B, T, static_cast<int>(n), C
    );

    // 2. 计算 dH: [B, T, n]
    dim3 blocks_dh(B * T, n);
    mhc::stream_distribute_bwd_dh_kernel<256><<<blocks_dh, threads, 0, stream>>>(
        d_H_post->typed_data(),
        reinterpret_cast<const mhc::floatX*>(grad.typed_data()),
        reinterpret_cast<const mhc::floatX*>(inp.typed_data()),
        B, T, static_cast<int>(n), C
    );

    return ffi::Error::Success();
}

// 注册 FFI 符号 (追加到文件末尾的 XLA_FFI_DEFINE_HANDLER_SYMBOL 序列中)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    StreamDistributeFwd, StreamDistributeFwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>() // inp
        .Arg<ffi::Buffer<ffi::F32>>()  // H_post
        .Ret<ffi::Buffer<ffi::BF16>>() // out
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    StreamDistributeBwd, StreamDistributeBwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>() // grad
        .Arg<ffi::Buffer<ffi::BF16>>() // inp
        .Arg<ffi::Buffer<ffi::F32>>()  // H_post
        .Ret<ffi::Buffer<ffi::BF16>>() // d_inp
        .Ret<ffi::Buffer<ffi::F32>>()  // d_H_post
);
/* -------------------- MHC Post-Op FFI -------------------- */

// 前向处理器
static ffi::Error MhcPostOpFwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> layer_out,  // [B, T, C]
    ffi::Buffer<ffi::BF16> x_expanded, // [B, T, n, C]
    ffi::Buffer<ffi::F32> H_post,      // [B, T, n]
    ffi::Buffer<ffi::F32> H_res,       // [B, T, n, n]
    ffi::ResultBuffer<ffi::BF16> out   // [B, T, n, C]
) {
    auto dims = x_expanded.dimensions();
    int64_t B = dims[0], T = dims[1], n = dims[2], C = dims[3];

    mhc::mhc_post_op_forward(
        reinterpret_cast<mhc::floatX*>(out->typed_data()),
        reinterpret_cast<const mhc::floatX*>(layer_out.typed_data()),
        reinterpret_cast<const mhc::floatX*>(x_expanded.typed_data()),
        H_post.typed_data(),
        H_res.typed_data(),
        B, T, static_cast<int>(n), C, stream
    );
    return ffi::Error::Success();
}
// 反向处理器
static ffi::Error MhcPostOpBwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> grad,       // [B, T, n, C]
    ffi::Buffer<ffi::BF16> layer_out,
    ffi::Buffer<ffi::BF16> x_expanded,
    ffi::Buffer<ffi::F32> H_post,
    ffi::Buffer<ffi::F32> H_res,
    ffi::ResultBuffer<ffi::BF16> d_layer_out,
    ffi::ResultBuffer<ffi::BF16> d_x_expanded,
    ffi::ResultBuffer<ffi::F32> d_H_post, // <--- 需要清零
    ffi::ResultBuffer<ffi::F32> d_H_res   // <--- 需要清零
) {
    auto dims = x_expanded.dimensions();
    int64_t B = dims[0], T = dims[1], n = dims[2], C = dims[3];

    // -----------------------------------------------------------------
    // 【关键修复】: 显式清零 Accumulation Buffer
    // 因为 Kernel 内部使用 atomicAdd，而 JAX 分配的显存包含垃圾数据
    // -----------------------------------------------------------------
    size_t size_h_post = B * T * n * sizeof(float);
    size_t size_h_res = B * T * n * n * sizeof(float);

    cudaMemsetAsync(d_H_post->typed_data(), 0, size_h_post, stream);
    cudaMemsetAsync(d_H_res->typed_data(), 0, size_h_res, stream);

    // 调用 Kernel
    mhc::mhc_post_op_backward_full(
        reinterpret_cast<mhc::floatX*>(d_layer_out->typed_data()),
        reinterpret_cast<mhc::floatX*>(d_x_expanded->typed_data()),
        d_H_post->typed_data(),
        d_H_res->typed_data(),
        reinterpret_cast<const mhc::floatX*>(grad.typed_data()),
        reinterpret_cast<const mhc::floatX*>(layer_out.typed_data()),
        reinterpret_cast<const mhc::floatX*>(x_expanded.typed_data()),
        H_post.typed_data(),
        H_res.typed_data(),
        B, T, static_cast<int>(n), C, stream
    );
    
    return ffi::Error::Success();
}
// --- 注册符号 ---
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhcPostOpFwd, MhcPostOpFwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>() 
        .Arg<ffi::Buffer<ffi::BF16>>() 
        .Arg<ffi::Buffer<ffi::F32>>()  
        .Arg<ffi::Buffer<ffi::F32>>()  
        .Ret<ffi::Buffer<ffi::BF16>>() 
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhcPostOpBwd, MhcPostOpBwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>() // grad
        .Arg<ffi::Buffer<ffi::BF16>>() // lo
        .Arg<ffi::Buffer<ffi::BF16>>() // xe
        .Arg<ffi::Buffer<ffi::F32>>()  // hp
        .Arg<ffi::Buffer<ffi::F32>>()  // hr
        .Ret<ffi::Buffer<ffi::BF16>>() // d_lo
        .Ret<ffi::Buffer<ffi::BF16>>() // d_xe
        .Ret<ffi::Buffer<ffi::F32>>()  // d_hp
        .Ret<ffi::Buffer<ffi::F32>>()  // d_hr
);

/* -------------------- MHC Pre-Op FFI -------------------- */

// 前向处理器：融合 Aggregate + Sigmoid + Sinkhorn 投影
static ffi::Error MhcPreOpFwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> x_expanded,  // [B, T, n, C]
    ffi::Buffer<ffi::F32> h_pre_raw,    // [B, T, n]
    ffi::Buffer<ffi::F32> h_post_raw,   // [B, T, n]
    ffi::Buffer<ffi::F32> h_res_raw,    // [B, T, n, n]
    ffi::ResultBuffer<ffi::BF16> x_layer_in, // [B, T, C]
    ffi::ResultBuffer<ffi::F32> H_pre,       // [B, T, n] (sigmoid后)
    ffi::ResultBuffer<ffi::F32> H_post,      // [B, T, n] (2*sigmoid后)
    ffi::ResultBuffer<ffi::F32> H_res,       // [B, T, n, n] (Sinkhorn后)
    std::int32_t sinkhorn_iters,
    float eps
) {
    auto dims = x_expanded.dimensions();
    int64_t B = dims[0];
    int64_t T = dims[1];
    int n = static_cast<int>(dims[2]);
    int64_t C = dims[3];
    
    // 调用 .cuh 中的融合前向接口
    mhc::mhc_pre_op_forward(
        reinterpret_cast<mhc::floatX*>(x_layer_in->typed_data()),
        H_pre->typed_data(),
        H_post->typed_data(),
        H_res->typed_data(),
        reinterpret_cast<const mhc::floatX*>(x_expanded.typed_data()),
        h_pre_raw.typed_data(),
        h_post_raw.typed_data(),
        h_res_raw.typed_data(),
        B, T, n, C, sinkhorn_iters, eps, stream
    );
    
    return ffi::Error::Success();
}

// 反向处理器：全量梯度回传（含 Sinkhorn 反向）
static ffi::Error MhcPreOpBwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> grad_layer_in,  // [B, T, C]
    ffi::Buffer<ffi::F32> grad_H_post,     // [B, T, n]
    ffi::Buffer<ffi::F32> grad_H_res,      // [B, T, n, n]
    ffi::Buffer<ffi::BF16> x_expanded,     // [B, T, n, C] (前向输入)
    ffi::Buffer<ffi::F32> H_pre,           // [B, T, n] (前向输出)
    ffi::Buffer<ffi::F32> H_post,          // [B, T, n] (前向输出)
    ffi::Buffer<ffi::F32> H_res_out,       // [B, T, n, n] (Sinkhorn后)
    ffi::Buffer<ffi::F32> h_res_raw,       // [B, T, n, n] (原始输入)
    ffi::ResultBuffer<ffi::BF16> d_x_expanded, // [B, T, n, C]
    ffi::ResultBuffer<ffi::F32> d_h_pre_raw,   // [B, T, n]
    ffi::ResultBuffer<ffi::F32> d_h_post_raw,  // [B, T, n]
    ffi::ResultBuffer<ffi::F32> d_h_res_raw,   // [B, T, n, n]
    std::int32_t sinkhorn_iters,
    float eps
) {
    auto dims = x_expanded.dimensions();
    int64_t B = dims[0];
    int64_t T = dims[1];
    int n = static_cast<int>(dims[2]);
    int64_t C = dims[3];

    // -----------------------------------------------------------------
    // 【关键修复】: 显式清零所有输出梯度缓冲区
    // PyTorch 版本使用 torch.zeros_like，FFI 侧需手动 Memset
    // 原因：1) 对齐框架行为；2) 防止未初始化数据导致的数值误差
    // -----------------------------------------------------------------
    size_t size_h_pre = B * T * n * sizeof(float);
    size_t size_h_post = B * T * n * sizeof(float);
    size_t size_h_res = B * T * n * n * sizeof(float);
    // d_x_expanded 由每个线程独占写入，无需清零

    cudaMemsetAsync(d_h_pre_raw->typed_data(), 0, size_h_pre, stream);
    cudaMemsetAsync(d_h_post_raw->typed_data(), 0, size_h_post, stream);
    cudaMemsetAsync(d_h_res_raw->typed_data(), 0, size_h_res, stream);

    // 调用 .cuh 中的融合反向接口
    mhc::mhc_pre_op_backward(
        reinterpret_cast<mhc::floatX*>(d_x_expanded->typed_data()),
        d_h_pre_raw->typed_data(),
        d_h_post_raw->typed_data(),
        d_h_res_raw->typed_data(),
        reinterpret_cast<const mhc::floatX*>(grad_layer_in.typed_data()),
        grad_H_post.typed_data(),
        grad_H_res.typed_data(),
        reinterpret_cast<const mhc::floatX*>(x_expanded.typed_data()),
        H_pre.typed_data(),
        H_post.typed_data(),
        H_res_out.typed_data(),
        h_res_raw.typed_data(),
        B, T, n, C, sinkhorn_iters, eps, stream
    );
    
    return ffi::Error::Success();
}

// 注册 FFI 符号（追加到文件末尾）
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhcPreOpFwd, MhcPreOpFwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()  // x_expanded
        .Arg<ffi::Buffer<ffi::F32>>()  // h_pre_raw
        .Arg<ffi::Buffer<ffi::F32>>()  // h_post_raw
        .Arg<ffi::Buffer<ffi::F32>>()  // h_res_raw
        .Ret<ffi::Buffer<ffi::BF16>>() // x_layer_in
        .Ret<ffi::Buffer<ffi::F32>>()  // H_pre
        .Ret<ffi::Buffer<ffi::F32>>()  // H_post
        .Ret<ffi::Buffer<ffi::F32>>()  // H_res
        .Attr<std::int32_t>("sinkhorn_iters")
        .Attr<float>("eps")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhcPreOpBwd, MhcPreOpBwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()  // grad_layer_in
        .Arg<ffi::Buffer<ffi::F32>>()  // grad_H_post
        .Arg<ffi::Buffer<ffi::F32>>()  // grad_H_res
        .Arg<ffi::Buffer<ffi::BF16>>() // x_expanded
        .Arg<ffi::Buffer<ffi::F32>>()  // H_pre
        .Arg<ffi::Buffer<ffi::F32>>()  // H_post
        .Arg<ffi::Buffer<ffi::F32>>()  // H_res_out
        .Arg<ffi::Buffer<ffi::F32>>()  // h_res_raw
        .Ret<ffi::Buffer<ffi::BF16>>() // d_x_expanded
        .Ret<ffi::Buffer<ffi::F32>>()  // d_h_pre_raw
        .Ret<ffi::Buffer<ffi::F32>>()  // d_h_post_raw
        .Ret<ffi::Buffer<ffi::F32>>()  // d_h_res_raw
        .Attr<std::int32_t>("sinkhorn_iters")
        .Attr<float>("eps")
);
