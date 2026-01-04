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