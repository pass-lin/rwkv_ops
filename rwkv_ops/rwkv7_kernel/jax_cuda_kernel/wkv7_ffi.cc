#include <xla/ffi/api/ffi.h>
#include <cuda_runtime.h>

extern "C" {
void cuda_forward(int B, int T, int H,
                  const void* w, const void* q,
                  const void* k, const void* v,
                  const void* z, const void* a,   // z 对应你 CUDA 里的 bf*z_
                  void* y,
                  float* s, float* sa, const float* h0);

void cuda_backward(int B, int T, int H,
                   const void* w, const void* q,
                   const void* k, const void* v,
                   const void* z, const void* a, const void* dy,
                   const float* s, const float* sa, const float* dht,
                   float* dh0,
                   void* dw, void* dq, void* dk, void* dv, void* dz, void* da);
}

namespace ffi = xla::ffi;

/* ---------- 前向 ---------- */
ffi::Error Wkv7FwdImpl(cudaStream_t stream,
                       ffi::Buffer<ffi::BF16> w,
                       ffi::Buffer<ffi::BF16> q,
                       ffi::Buffer<ffi::BF16> k,
                       ffi::Buffer<ffi::BF16> v,
                       ffi::Buffer<ffi::BF16> z,   // 对应 CUDA 里的 z
                       ffi::Buffer<ffi::BF16> a,
                       ffi::Buffer<ffi::F32> h0,
                       ffi::ResultBuffer<ffi::BF16> y,
                       ffi::ResultBuffer<ffi::F32> s,
                       ffi::ResultBuffer<ffi::F32> sa) {
  auto dims = w.dimensions();
  int B = dims[0], T = dims[1], H = dims[2];
  constexpr int C = 64;  // 与 CUDA 侧 _C_ 保持一致

  cuda_forward(B, T, H,
               reinterpret_cast<const void*>(w.typed_data()),
               reinterpret_cast<const void*>(q.typed_data()),
               reinterpret_cast<const void*>(k.typed_data()),
               reinterpret_cast<const void*>(v.typed_data()),
               reinterpret_cast<const void*>(z.typed_data()),
               reinterpret_cast<const void*>(a.typed_data()),
               y->typed_data(),
               s->typed_data(),
               sa->typed_data(),
               h0.typed_data());
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Wkv7Fwd, Wkv7FwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()  // w
        .Arg<ffi::Buffer<ffi::BF16>>()  // q
        .Arg<ffi::Buffer<ffi::BF16>>()  // k
        .Arg<ffi::Buffer<ffi::BF16>>()  // v
        .Arg<ffi::Buffer<ffi::BF16>>()  // z
        .Arg<ffi::Buffer<ffi::BF16>>()  // a
        .Arg<ffi::Buffer<ffi::F32>>()   // h0
        .Ret<ffi::Buffer<ffi::BF16>>()  // y
        .Ret<ffi::Buffer<ffi::F32>>()   // s
        .Ret<ffi::Buffer<ffi::F32>>()   // sa
);

/* ---------- 反向 ---------- */
ffi::Error Wkv7BwdImpl(cudaStream_t stream,
                       ffi::Buffer<ffi::BF16> w,
                       ffi::Buffer<ffi::BF16> q,
                       ffi::Buffer<ffi::BF16> k,
                       ffi::Buffer<ffi::BF16> v,
                       ffi::Buffer<ffi::BF16> z,
                       ffi::Buffer<ffi::BF16> a,
                       ffi::Buffer<ffi::BF16> dy,
                       ffi::Buffer<ffi::F32> s,
                       ffi::Buffer<ffi::F32> sa,
                       ffi::Buffer<ffi::F32> dht,
                       ffi::ResultBuffer<ffi::F32> dh0,
                       ffi::ResultBuffer<ffi::BF16> dw,
                       ffi::ResultBuffer<ffi::BF16> dq,
                       ffi::ResultBuffer<ffi::BF16> dk,
                       ffi::ResultBuffer<ffi::BF16> dv,
                       ffi::ResultBuffer<ffi::BF16> dz,
                       ffi::ResultBuffer<ffi::BF16> da) {
  auto dims = w.dimensions();
  int B = dims[0], T = dims[1], H = dims[2];
  constexpr int C = 64;

  cuda_backward(B, T, H,
                reinterpret_cast<const void*>(w.typed_data()),
                reinterpret_cast<const void*>(q.typed_data()),
                reinterpret_cast<const void*>(k.typed_data()),
                reinterpret_cast<const void*>(v.typed_data()),
                reinterpret_cast<const void*>(z.typed_data()),
                reinterpret_cast<const void*>(a.typed_data()),
                reinterpret_cast<const void*>(dy.typed_data()),
                s.typed_data(),
                sa.typed_data(),
                dht.typed_data(),
                dh0->typed_data(),
                reinterpret_cast<void*>(dw->typed_data()),
                reinterpret_cast<void*>(dq->typed_data()),
                reinterpret_cast<void*>(dk->typed_data()),
                reinterpret_cast<void*>(dv->typed_data()),
                reinterpret_cast<void*>(dz->typed_data()),
                reinterpret_cast<void*>(da->typed_data()));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Wkv7Bwd, Wkv7BwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()   // w
        .Arg<ffi::Buffer<ffi::BF16>>()   // q
        .Arg<ffi::Buffer<ffi::BF16>>()   // k
        .Arg<ffi::Buffer<ffi::BF16>>()   // v
        .Arg<ffi::Buffer<ffi::BF16>>()   // z
        .Arg<ffi::Buffer<ffi::BF16>>()   // a
        .Arg<ffi::Buffer<ffi::BF16>>()   // dy
        .Arg<ffi::Buffer<ffi::F32>>()    // s
        .Arg<ffi::Buffer<ffi::F32>>()    // sa
        .Arg<ffi::Buffer<ffi::F32>>()    // dht
        .Ret<ffi::Buffer<ffi::F32>>()    // dh0
        .Ret<ffi::Buffer<ffi::BF16>>()   // dw
        .Ret<ffi::Buffer<ffi::BF16>>()   // dq
        .Ret<ffi::Buffer<ffi::BF16>>()   // dk
        .Ret<ffi::Buffer<ffi::BF16>>()   // dv
        .Ret<ffi::Buffer<ffi::BF16>>()   // dz
        .Ret<ffi::Buffer<ffi::BF16>>()   // da
);