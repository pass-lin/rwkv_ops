#include <stdio.h>
#include <assert.h>
#include "xla/ffi/api/ffi.h"
#include "cuda_bf16.h"

namespace ffi = xla::ffi;
typedef __nv_bfloat16 bf16;

//----------------------------------------------------------------------------//
//                            Forward pass                                    //
//----------------------------------------------------------------------------//
#define _CHUNK_LEN_ 16
#define _C_ 64

template <typename F>
__global__ void kernel_forward(int T, int H,
			       F* w_, F* q_, F* k_, F* v_, F* a_, F* b_,
			       F* in_s_, uint32_t* length_, bool* new_starts_,
			       F* y_, float* s_, float* sa_, F* out_s_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    const uint32_t length = length_[bb];
    bool* new_starts = new_starts_ + bb * T;

    int state_ind = bb * H * C * C + hh * C * C + i * C;
    
    float state[C];
    float out_state[C];
    #pragma unroll
    for (int j = 0; j < C; j++) {
      state[j] = in_s_[state_ind + j];
    }
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    for (int t = 0; t < T; t++) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = float(q_[ind]);
        w[i] = __expf(-__expf(float(w_[ind])));
        k[i] = float(k_[ind]);
        a[i] = float(a_[ind]);
        b[i] = float(b_[ind]);
        __syncthreads();

	if (new_starts[t]) {
	  #pragma unroll
	  for (int j = 0; j < C; j++) {
	    state[j] = 0;
	  }
	}

        float sa = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            sa += a[j] * state[j];
        }
        sa_[ind] = sa;

        float v = float(v_[ind]);
        float y = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            float& s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v;
	    out_state[j] = t < length ? s : out_state[j];
            y += s * q[j];
        }
        y_[ind] = F(y);

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i;
#pragma unroll
            for (int j = 0; j < C; j++) {
	      s_[base + j*C] = state[j];
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < C; j++) {
      out_s_[state_ind + j] = out_state[j];
    }
}

template <typename T, typename Q>
ffi::Error cuda_forward_typed(cudaStream_t stream,
			      T w, T q, T k, T v, T a, T b,
			      T in_s, ffi::Buffer<ffi::U32> length, ffi::Buffer<ffi::PRED> new_starts,
			      ffi::Result<T> y, ffi::Result<ffi::Buffer<ffi::F32>> s, ffi::Result<ffi::Buffer<ffi::F32>> sa, ffi::Result<T> out_s,
			      size_t B, size_t T_, size_t H) {
  kernel_forward<<<dim3(H,B), dim3(_C_), 0, stream>>>(
					   T_, H,
					   (Q*)w.typed_data(),(Q*)q.typed_data(),(Q*)k.typed_data(),(Q*)v.typed_data(),(Q*)a.typed_data(),(Q*)b.typed_data(),
					   (Q*)in_s.typed_data(), length.typed_data(), new_starts.typed_data(),
					   (Q*)y->typed_data(),(float*)s->typed_data(),(float*)sa->typed_data(),(Q*)out_s->typed_data());
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }

  return ffi::Error::Success();
}

ffi::Error cuda_forward_f32(cudaStream_t stream,
			      ffi::Buffer<ffi::F32> w, ffi::Buffer<ffi::F32> q, ffi::Buffer<ffi::F32> k, ffi::Buffer<ffi::F32> v, ffi::Buffer<ffi::F32> a, ffi::Buffer<ffi::F32> b,
			      ffi::Buffer<ffi::F32> in_s, ffi::Buffer<ffi::U32> length, ffi::Buffer<ffi::PRED> new_starts,
			      ffi::Result<ffi::Buffer<ffi::F32>> y, ffi::Result<ffi::Buffer<ffi::F32>> s, ffi::Result<ffi::Buffer<ffi::F32>> sa,ffi::Result<ffi::Buffer<ffi::F32>> out_s,
			      size_t B, size_t T_, size_t H) {
  return cuda_forward_typed<ffi::Buffer<ffi::F32>, float>(stream,
							  w, q, k, v, a, b,
							  in_s, length, new_starts,
							  y, s, sa, out_s,
							  B, T_, H);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    WKV7FwdF32, cuda_forward_f32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::F32>>()              // w
        .Arg<ffi::Buffer<ffi::F32>>()              // q
        .Arg<ffi::Buffer<ffi::F32>>()              // k
        .Arg<ffi::Buffer<ffi::F32>>()              // v
        .Arg<ffi::Buffer<ffi::F32>>()              // a
        .Arg<ffi::Buffer<ffi::F32>>()              // b
        .Arg<ffi::Buffer<ffi::F32>>()              // in_s
        .Arg<ffi::Buffer<ffi::U32>>()              // length
        .Arg<ffi::Buffer<ffi::PRED>>()              // new_starts
        .Ret<ffi::Buffer<ffi::F32>>()              // y
        .Ret<ffi::Buffer<ffi::F32>>()              // s
        .Ret<ffi::Buffer<ffi::F32>>()              // sa
        .Ret<ffi::Buffer<ffi::F32>>()              // out_s
    .Attr<size_t>("B")
    .Attr<size_t>("T")
    .Attr<size_t>("H"),
    {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled


ffi::Error cuda_forward_bf16(cudaStream_t stream,
			      ffi::Buffer<ffi::BF16> w, ffi::Buffer<ffi::BF16> q, ffi::Buffer<ffi::BF16> k, ffi::Buffer<ffi::BF16> v, ffi::Buffer<ffi::BF16> a, ffi::Buffer<ffi::BF16> b,
			      ffi::Buffer<ffi::BF16> in_s, ffi::Buffer<ffi::U32> length, ffi::Buffer<ffi::PRED> new_starts,
			      ffi::Result<ffi::Buffer<ffi::BF16>> y, ffi::Result<ffi::Buffer<ffi::F32>> s, ffi::Result<ffi::Buffer<ffi::F32>> sa,ffi::Result<ffi::Buffer<ffi::BF16>> out_s,
			      size_t B, size_t T_, size_t H) {
  return cuda_forward_typed<ffi::Buffer<ffi::BF16>, bf16>(stream,
							  w, q, k, v, a, b,
							  in_s, length, new_starts,
							  y, s, sa, out_s,
							  B, T_, H);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    WKV7FwdBF16, cuda_forward_bf16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::BF16>>()              // w
        .Arg<ffi::Buffer<ffi::BF16>>()              // q
        .Arg<ffi::Buffer<ffi::BF16>>()              // k
        .Arg<ffi::Buffer<ffi::BF16>>()              // v
        .Arg<ffi::Buffer<ffi::BF16>>()              // a
        .Arg<ffi::Buffer<ffi::BF16>>()              // b
        .Arg<ffi::Buffer<ffi::BF16>>()              // in_s
        .Arg<ffi::Buffer<ffi::U32>>()              // length
        .Arg<ffi::Buffer<ffi::PRED>>()              // new_starts
        .Ret<ffi::Buffer<ffi::BF16>>()              // y
        .Ret<ffi::Buffer<ffi::F32>>()              // s
        .Ret<ffi::Buffer<ffi::F32>>()              // sa
        .Ret<ffi::Buffer<ffi::BF16>>()              // out_s
    .Attr<size_t>("B")
    .Attr<size_t>("T")
    .Attr<size_t>("H"),
    {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled






//----------------------------------------------------------------------------//
//                            Backward pass                                   //
//----------------------------------------------------------------------------//



template <typename F>
__global__ void kernel_backward(int T, int H,
				F* w_, F* q_, F* k_, F* v_, F* a_, F* b_,
				F* in_s_, uint32_t* length_, bool* new_starts_,
				F* dy_, float * __restrict__ s_, float * __restrict__ sa_, F* ds_out_,
				F* dw_, F* dq_, F* dk_, F* dv_, F* da_, F* db_, F* ds_in_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    int state_ind = bb * H * C * C + hh * C * C + i * C;


    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; t--) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = qi = float(q_[ind]);
        float wi_fac = -__expf(float(w_[ind]));
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = float(k_[ind]);
        a[i] = ai = float(a_[ind]);
        b[i] = bi = float(b_[ind]);
        v[i] = float(v_[ind]);
        dy[i] = dyi = float(dy_[ind]);
        sa[i] = sa_[ind];
        __syncthreads();

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C;
#pragma unroll
            for (int j = 0; j < C; j++) {
                stateT[j] = s_[base + j];
            }
        }

        float dq = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dq += stateT[j]*dy[j];
        }
        dq_[ind] = F(dq);

        float iwi = 1.0f/wi;
#pragma unroll        
        for (int j = 0; j < C; j++) {
            stateT[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }

        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j]*stateT[j];
            dk += dstateT[j]*v[j];
            dv += dstate[j]*k[j];
            dSb += dstate[j]*b[j];
            db += dstateT[j]*sa[j];
        }
        dw_[ind] = F(dw * wi * wi_fac);
        dk_[ind] = F(dk);
        dv_[ind] = F(dv);
        db_[ind] = F(db);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            da += stateT[j]*dSb_shared[j];
        }
        da_[ind] = F(da);

#pragma unroll        
        for (int j = 0; j < C; j++) {
            dstate[j] = dstate[j]*w[j] + dSb * a[j];
            dstateT[j] = dstateT[j]*wi + ai * dSb_shared[j];
        }
    }

    #pragma unroll
    for (int j = 0; j < C; j++) {
      ds_in_[state_ind + j] = F(0);
    }
}

template <typename T, typename Q>
ffi::Error cuda_backward_typed(cudaStream_t stream,
			      T w, T q, T k, T v, T a, T b,
			      T in_s, ffi::Buffer<ffi::U32> length, ffi::Buffer<ffi::PRED> new_starts,
			       T dy, ffi::Buffer<ffi::F32> s, ffi::Buffer<ffi::F32> sa, T ds_out,
			      ffi::Result<T> dw, ffi::Result<T> dq, ffi::Result<T> dk, ffi::Result<T> dv, ffi::Result<T> da, ffi::Result<T> db, ffi::Result<T> ds_in, 
			      size_t B, size_t T_, size_t H) {
  kernel_backward<<<dim3(H,B), dim3(_C_), 0, stream>>>(
					   T_, H,
					   (Q*)w.typed_data(),(Q*)q.typed_data(),(Q*)k.typed_data(),(Q*)v.typed_data(),(Q*)a.typed_data(),(Q*)b.typed_data(),
					   (Q*)in_s.typed_data(), length.typed_data(), new_starts.typed_data(),
					   (Q*)dy.typed_data(), (float*)s.typed_data(), (float*)sa.typed_data(),(Q*)ds_out.typed_data(),
					   (Q*)dw->typed_data(),(Q*)dq->typed_data(),(Q*)dk->typed_data(),(Q*)dv->typed_data(),(Q*)da->typed_data(),(Q*)db->typed_data(),(Q*)ds_in->typed_data());
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }

  return ffi::Error::Success();
}


ffi::Error cuda_backward_f32(cudaStream_t stream,
			     ffi::Buffer<ffi::F32> w, ffi::Buffer<ffi::F32> q, ffi::Buffer<ffi::F32> k, ffi::Buffer<ffi::F32> v, ffi::Buffer<ffi::F32> a, ffi::Buffer<ffi::F32> b,
			      ffi::Buffer<ffi::F32> in_s, ffi::Buffer<ffi::U32> length, ffi::Buffer<ffi::PRED> new_starts,
			       ffi::Buffer<ffi::F32> dy, ffi::Buffer<ffi::F32> s, ffi::Buffer<ffi::F32> sa, ffi::Buffer<ffi::F32> ds_out, 
			      ffi::Result<ffi::Buffer<ffi::F32>> dw, ffi::Result<ffi::Buffer<ffi::F32>> dq, ffi::Result<ffi::Buffer<ffi::F32>> dk, ffi::Result<ffi::Buffer<ffi::F32>> dv, ffi::Result<ffi::Buffer<ffi::F32>> da, ffi::Result<ffi::Buffer<ffi::F32>> db, ffi::Result<ffi::Buffer<ffi::F32>> ds_in, 
			      size_t B, size_t T_, size_t H) {
  return cuda_backward_typed<ffi::Buffer<ffi::F32>, float>(
							   stream,
							   w,q,k,v,a,b,
							   in_s,length,new_starts,
							   dy,s,sa,ds_out,
							   dw,dq,dk,dv,da,db,ds_in,
							   B,T_,H
);
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    WKV7BwdF32, cuda_backward_f32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::F32>>()              // w
        .Arg<ffi::Buffer<ffi::F32>>()              // q
        .Arg<ffi::Buffer<ffi::F32>>()              // k
        .Arg<ffi::Buffer<ffi::F32>>()              // v
        .Arg<ffi::Buffer<ffi::F32>>()              // a
        .Arg<ffi::Buffer<ffi::F32>>()              // b
        .Arg<ffi::Buffer<ffi::F32>>()              // in_s
        .Arg<ffi::Buffer<ffi::U32>>()              // length
        .Arg<ffi::Buffer<ffi::PRED>>()              // new_starts
        .Arg<ffi::Buffer<ffi::F32>>()              // dy
        .Arg<ffi::Buffer<ffi::F32>>()              // s
        .Arg<ffi::Buffer<ffi::F32>>()              // sa
        .Arg<ffi::Buffer<ffi::F32>>()              // ds_out
        .Ret<ffi::Buffer<ffi::F32>>()              // dw
        .Ret<ffi::Buffer<ffi::F32>>()              // dq
        .Ret<ffi::Buffer<ffi::F32>>()              // dk
        .Ret<ffi::Buffer<ffi::F32>>()              // dv
        .Ret<ffi::Buffer<ffi::F32>>()              // da
        .Ret<ffi::Buffer<ffi::F32>>()              // db
        .Ret<ffi::Buffer<ffi::F32>>()              // ds_in
    .Attr<size_t>("B")
    .Attr<size_t>("T")
    .Attr<size_t>("H"),
    {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled



ffi::Error cuda_backward_bf16(cudaStream_t stream,
			     ffi::Buffer<ffi::BF16> w, ffi::Buffer<ffi::BF16> q, ffi::Buffer<ffi::BF16> k, ffi::Buffer<ffi::BF16> v, ffi::Buffer<ffi::BF16> a, ffi::Buffer<ffi::BF16> b,
			      ffi::Buffer<ffi::BF16> in_s, ffi::Buffer<ffi::U32> length, ffi::Buffer<ffi::PRED> new_starts,
			       ffi::Buffer<ffi::BF16> dy, ffi::Buffer<ffi::F32> s, ffi::Buffer<ffi::F32> sa, ffi::Buffer<ffi::BF16> ds_out, 
			      ffi::Result<ffi::Buffer<ffi::BF16>> dw, ffi::Result<ffi::Buffer<ffi::BF16>> dq, ffi::Result<ffi::Buffer<ffi::BF16>> dk, ffi::Result<ffi::Buffer<ffi::BF16>> dv, ffi::Result<ffi::Buffer<ffi::BF16>> da, ffi::Result<ffi::Buffer<ffi::BF16>> db, ffi::Result<ffi::Buffer<ffi::BF16>> ds_in, 
			      size_t B, size_t T_, size_t H) {
  return cuda_backward_typed<ffi::Buffer<ffi::BF16>, bf16>(
							   stream,
							   w,q,k,v,a,b,
							   in_s,length,new_starts,
							   dy,s,sa,ds_out,
							   dw,dq,dk,dv,da,db,ds_in,
							   B,T_,H
);
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    WKV7BwdBF16, cuda_backward_bf16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::BF16>>()              // w
        .Arg<ffi::Buffer<ffi::BF16>>()              // q
        .Arg<ffi::Buffer<ffi::BF16>>()              // k
        .Arg<ffi::Buffer<ffi::BF16>>()              // v
        .Arg<ffi::Buffer<ffi::BF16>>()              // a
        .Arg<ffi::Buffer<ffi::BF16>>()              // b
        .Arg<ffi::Buffer<ffi::BF16>>()              // in_s
        .Arg<ffi::Buffer<ffi::U32>>()              // length
        .Arg<ffi::Buffer<ffi::PRED>>()              // new_starts
        .Arg<ffi::Buffer<ffi::BF16>>()              // dy
        .Arg<ffi::Buffer<ffi::F32>>()              // s
        .Arg<ffi::Buffer<ffi::F32>>()              // sa
        .Arg<ffi::Buffer<ffi::BF16>>()              // ds_out
        .Ret<ffi::Buffer<ffi::BF16>>()              // dw
        .Ret<ffi::Buffer<ffi::BF16>>()              // dq
        .Ret<ffi::Buffer<ffi::BF16>>()              // dk
        .Ret<ffi::Buffer<ffi::BF16>>()              // dv
        .Ret<ffi::Buffer<ffi::BF16>>()              // da
        .Ret<ffi::Buffer<ffi::BF16>>()              // db
        .Ret<ffi::Buffer<ffi::BF16>>()              // ds_in
    .Attr<size_t>("B")
    .Attr<size_t>("T")
    .Attr<size_t>("H"),
    {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled






// void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa) {
//     forward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,y,s,sa);
// }



// #include <cuda_bf16.h>
// #include <assert.h>

// using bf = __nv_bfloat16;
// __device__ inline float to_float(const bf & u) { return __bfloat162float(u); }
// __device__ inline bf to_bf(const float & u) { return __float2bfloat16_rn(u); }

// typedef bf * __restrict__ F_;

// __global__ void forward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, bf* y_, float* s_, float* sa_) {
//     constexpr int C = _C_;
//     int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

//     float state[C] = {0};
//     __shared__ float q[C], k[C], w[C], a[C], b[C];

//     for (int t = 0; t < T; t++) {
//         int ind = bb*T*H*C + t*H*C + hh * C + i;
//         __syncthreads();
//         q[i] = to_float(q_[ind]);
//         w[i] = __expf(-__expf(to_float(w_[ind])));
//         k[i] = to_float(k_[ind]);
//         a[i] = to_float(a_[ind]);
//         b[i] = to_float(b_[ind]);
//         __syncthreads();

//         float sa = 0;
// #pragma unroll
//         for (int j = 0; j < C; j++) {
//             sa += a[j] * state[j];
//         }
//         sa_[ind] = sa;

//         float v = to_float(v_[ind]);
//         float y = 0;
// #pragma unroll
//         for (int j = 0; j < C; j++) {
//             float& s = state[j];
//             s = s * w[j] + sa * b[j] + k[j] * v;
//             y += s * q[j];
//         }
//         y_[ind] = to_bf(y);

//         if ((t+1)%_CHUNK_LEN_ == 0) {
//             int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i;
// #pragma unroll
//             for (int j = 0; j < C; j++) {
//                 s_[base + j*C] = state[j];
//             }
//         }
//     }
// }

// __global__ void backward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, float * __restrict__ s_, float * __restrict__ sa_, bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_) {
//     constexpr int C = _C_;
//     int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

//     float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
//     __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
//     float qi, wi, ki, ai, bi, dyi;

//     for (int t = T-1; t >= 0; t--) {
//         int ind = bb*T*H*C + t*H*C + hh * C + i;
//         __syncthreads();
//         q[i] = qi = to_float(q_[ind]);
//         float wi_fac = -__expf(to_float(w_[ind]));
//         w[i] = wi = __expf(wi_fac);
//         k[i] = ki = to_float(k_[ind]);
//         a[i] = ai = to_float(a_[ind]);
//         b[i] = bi = to_float(b_[ind]);
//         v[i] = to_float(v_[ind]);
//         dy[i] = dyi = to_float(dy_[ind]);
//         sa[i] = sa_[ind];
//         __syncthreads();

//         if ((t+1)%_CHUNK_LEN_ == 0) {
//             int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C;
// #pragma unroll
//             for (int j = 0; j < C; j++) {
//                 stateT[j] = s_[base + j];
//             }
//         }

//         float dq = 0;
// #pragma unroll
//         for (int j = 0; j < C; j++) {
//             dq += stateT[j]*dy[j];
//         }
//         dq_[ind] = to_bf(dq);

//         float iwi = 1.0f/wi;
// #pragma unroll        
//         for (int j = 0; j < C; j++) {
//             stateT[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
//             dstate[j] += dyi * q[j];
//             dstateT[j] += qi * dy[j];
//         }

//         float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
// #pragma unroll
//         for (int j = 0; j < C; j++) {
//             dw += dstateT[j]*stateT[j];
//             dk += dstateT[j]*v[j];
//             dv += dstate[j]*k[j];
//             dSb += dstate[j]*b[j];
//             db += dstateT[j]*sa[j];
//         }
//         dw_[ind] = to_bf(dw * wi * wi_fac);
//         dk_[ind] = to_bf(dk);
//         dv_[ind] = to_bf(dv);
//         db_[ind] = to_bf(db);

//         __syncthreads();
//         dSb_shared[i] = dSb;
//         __syncthreads();

//         float da = 0;
// #pragma unroll
//         for (int j = 0; j < C; j++) {
//             da += stateT[j]*dSb_shared[j];
//         }
//         da_[ind] = to_bf(da);

// #pragma unroll        
//         for (int j = 0; j < C; j++) {
//             dstate[j] = dstate[j]*w[j] + dSb * a[j];
//             dstateT[j] = dstateT[j]*wi + ai * dSb_shared[j];
//         }
//     }
// }

// void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa) {
//     forward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,y,s,sa);
// }
// void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da) {
//     assert(T%_CHUNK_LEN_ == 0);
//     backward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,dy,s,sa,dw,dq,dk,dv,dz,da);
// }
