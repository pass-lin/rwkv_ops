/* wkv7_ffi.cc
 * XLA-FFI 封装：把 CUDA kernel 暴露给 JAX
 * 注意：本文件不出现 __nv_bfloat16，全部用 void* 传递
 */
#include <xla/ffi/api/ffi.h>
#include <cuda_runtime.h>

extern "C" {
void wkv7_fwd_cuda(
        int B, int T, int H, int C,
        const void* w, const void* q,
        const void* k, const void* v,
        const void* a,const void* b,
        const float* h0,
        void* y, float* s, float* sa
                );
}

namespace ffi = xla::ffi;

ffi::Error Wkv7FwdImpl(cudaStream_t stream,
                       ffi::Buffer<ffi::BF16> w,
                       ffi::Buffer<ffi::BF16> q,
                       ffi::Buffer<ffi::BF16> k,
                       ffi::Buffer<ffi::BF16> v,
                       ffi::Buffer<ffi::BF16> a,
                       ffi::Buffer<ffi::BF16> b,
                       ffi::Buffer<ffi::F32> h0,
                       ffi::ResultBuffer<ffi::BF16> y,
                       ffi::ResultBuffer<ffi::F32> s,
                       ffi::ResultBuffer<ffi::F32> sa) {
    auto dims = w.dimensions();
    int B = dims[0], T = dims[1], H = dims[2], C = dims[3];
    wkv7_fwd_cuda(B, T, H, C,
                  w.typed_data(), q.typed_data(), k.typed_data(),
                  v.typed_data(),  a.typed_data(), b.typed_data(),
                  h0.typed_data(),
                  y->typed_data(), s->typed_data(), sa->typed_data());
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Wkv7Fwd, Wkv7FwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()   // w
        .Arg<ffi::Buffer<ffi::BF16>>()   // q
        .Arg<ffi::Buffer<ffi::BF16>>()   // k
        .Arg<ffi::Buffer<ffi::BF16>>()   // v
        .Arg<ffi::Buffer<ffi::BF16>>()   // a
        .Arg<ffi::Buffer<ffi::BF16>>()   // b
        .Arg<ffi::Buffer<ffi::F32>>()    // h0
        .Ret<ffi::Buffer<ffi::BF16>>()   // y
        .Ret<ffi::Buffer<ffi::F32>>()    // s
        .Ret<ffi::Buffer<ffi::F32>>()    // sa
    , {ffi::Traits::kCmdBufferCompatible});