#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <xla/ffi/api/ffi.h>
#include <vector>
#include <cstdint>

// 公共头文件路径
#include "../common_kernel/include/mhc_types.h"
#include "../common_kernel/kernels/sinkhorn_knopp.cuh"

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