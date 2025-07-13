/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include <hip/hip_runtime.h>
#include "kernel_helpers.h"
#include "kernels.h"
#include "stdio.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <assert.h>
namespace {

#define DISPATCH_Vector_TYPES(TYPEIN, TYPEOUT,NAME, ...)                              \
  switch (TYPEIN) {                                                            \
  case gpu_ops::ElementType::F32: {                                            \
    using input_type = float;                                                  \
    switch (TYPEOUT) {                                                         \
      case gpu_ops::ElementType::F32: {                                          \
        using output_type = float;                                              \
        __VA_ARGS__;                                                             \
        break;                                                                   \
      }                                                                          \
      case gpu_ops::ElementType::F16: {                                          \
        using output_type = __half;                                             \
        __VA_ARGS__;                                                             \
        break;                                                                   \
      }                                                                          \
      case gpu_ops::ElementType::BF16: {                                         \
        using output_type = hip_bfloat16;                                      \
        __VA_ARGS__;                                                             \
        break;                                                                   \
      }                                                                          \
      default:                                                                   \
        break;                                                                   \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  case gpu_ops::ElementType::F16: {                                            \
    using input_type = __half;                                                \
    switch (TYPEOUT) {                                                         \
    case gpu_ops::ElementType::F32: {                                          \
      using output_type = float;                                              \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F16: {                                          \
      using output_type = __half;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::BF16: {                                         \
      using output_type = hip_bfloat16;                                      \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  case gpu_ops::ElementType::BF16: {                                           \
    using input_type = hip_bfloat16;                                          \
    switch (TYPEOUT) {                                                         \
    case gpu_ops::ElementType::F32: {                                          \
      using output_type = float;                                              \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F16: {                                          \
      using output_type = __half;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::BF16: {                                         \
      using output_type = hip_bfloat16;                                      \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  default:                                                                     \
    break;                                                                     \
  }


static_assert(_N_ % 4 ==0,"the size of head must be the times of 4.");





template <typename F_in,typename F_out>
__device__ void kernel_forward_core(const int B, const int T, const int C, const int H,const int b, const int h, const int i, const float* state,
                               const F_in *__restrict__ const _r, const F_in *__restrict__ const _k, const F_in *__restrict__ const _v, const F_in *__restrict__ _w, const F_in *__restrict__ _u,
                               F_out *__restrict__ const _y)
{

    _u += h*_N_;

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];
    //float state[_N_] = {0};

    __syncthreads();
    u[i] = float(_u[i]);
    __syncthreads();

    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();
        w[i] = __expf(-__expf(float(_w[t])));
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float v = float(_v[t]);
        float y = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j+=4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& w_ = (float4&)(w[j]);
            const float4& u_ = (float4&)(u[j]);
            float4& s = (float4&)(state[j]);
            float4 x;

            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;

            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }
        _y[t] = F_out(y);
    }
}




template <typename F_in,typename F_out>
__global__ void kernel_forward_state(const int B, const int T, const int C, const int H,const bool is_custom_state,const int32_t* map,
                               const F_in *__restrict__ const _r, const F_in *__restrict__ const _k, const F_in *__restrict__ const _v, const F_in *__restrict__ _w, const F_in *__restrict__ _u,
                               const F_out *__restrict__ _s, F_out *__restrict__ const _y, F_out *__restrict__ const _ys)
{
  const int b = blockIdx.x / H;
  const int h = blockIdx.x % H;
  const int i = threadIdx.x;
  float state[_N_] = {0};
  if(is_custom_state){  
    assert(map[b] >=0 && map[b] < B);

    const int64_t input_state_offset = map[b] * H * _N_ *_N_ + h * _N_ * _N_ + i;

    for(int j= 0; j< _N_; j++){
      state[j] = float(_s[j*_N_ + input_state_offset]);
    }
  }
  
  const int64_t current_state_offset = b * H * _N_ *_N_ + h * _N_ * _N_ + i;

  kernel_forward_core(B, T, C, H, b, h, i, state, _r, _k, _v, _w, _u, _y);
  for(int j=0; j< _N_; j++){
    _ys[j*_N_ + current_state_offset] = F_out(state[j]);
  }
}


template <typename F_in,typename F_out>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F_in *__restrict__ const _r, const F_in *__restrict__ const _k, const F_in *__restrict__ const _v, const F_in *__restrict__ _w, const F_in *__restrict__ _u,
                               F_out *__restrict__ const _y)
{
  const int b = blockIdx.x / H;
  const int h = blockIdx.x % H;
  const int i = threadIdx.x;
  float state[_N_] = {0};
  kernel_forward_core(B, T, C, H, b, h, i, state, _r, _k, _v, _w, _u, _y);
}

template <typename F_in, typename F_out>
__global__ void kernel_backward_101(const int B, const int T, const int C, const int H,
    const F_in *__restrict__ const _r, const F_in *__restrict__ const _k, const F_in *__restrict__ const _v, const F_in *__restrict__ _w,
    const F_in *__restrict__ _u, const F_out *__restrict__ const _gy,
    F_out *__restrict__ const _gr, F_out *__restrict__ const _gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float v[_N_], gy[_N_];

    const float u = float(_u[h*_N_ + i]);

    float state[_N_] = {0};

    const int t_0 = b*T*C + h*_N_ + i;
    const int t_T = t_0 + T*C;

    float gu = 0;
    for (int t = t_0; t < t_T; t += C)
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]);
        __syncthreads();

        const float k = float(_k[t]);
        const float w = __expf(-__expf(float(_w[t])));
        float gr = 0, gu_ = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            float x = k * v[j];

            gr += (u * x + s) * gy[j];
            gu_ += x * gy[j];
            s = s * w + x;
        }
        _gr[t] = F_out(gr);
        gu += float(_r[t]) * gu_;
    }
    _gu[b*C + h*_N_ + i] = F_out(gu);
}

template <typename F_in, typename F_out>
__global__ void kernel_backward_102(const int B, const int T, const int C, const int H,
    const F_in *__restrict__ const _r, const F_in *__restrict__ const _k, const F_in *__restrict__ const _v,
    const F_in *__restrict__ _w, const F_in *__restrict__ _u, const F_out *__restrict__ const _gy,
    F_out *__restrict__ const _gk)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float v[_N_], gy[_N_];

    const float u = float(_u[h*_N_ + i]);

    float scccc[_N_] = {0};

    const int t_0 = b*T*C + h*_N_ + i;
    const int t_T_1 = t_0 + (T-1)*C;

    for (int t = t_T_1; t >= t_0; t -= C)
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]);
        __syncthreads();

        const float rr = float(_r[t]);
        const float w = __expf(-__expf(float(_w[t])));
        float gk = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = scccc[j];
            float x = rr * gy[j];
            
            gk += (u * x + s) * v[j];
            s = x + s * w;
        }
        _gk[t] = F_out(gk);
    }
}

template <typename F_in, typename F_out>
__global__ void kernel_backward_103(const int B, const int T, const int C, const int H,
    const F_in *__restrict__ const _r, const F_in *__restrict__ const _k, const F_in *__restrict__ const _v,
    const F_in *__restrict__ _w, const F_in *__restrict__ _u, const F_out *__restrict__ const _gy,
    F_out *__restrict__ const _gv)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _u += h*_N_;

    __shared__ float u_[_N_], r[_N_], k[_N_], w_[_N_];
    __syncthreads();
    u_[i] = float(_u[i]);
    __syncthreads();

    float sdddd[_N_] = {0};

    const int t_0 = b*T*C + h*_N_ + i;
    const int t_T_1 = t_0 + (T-1)*C;

    for (int t = t_T_1; t >= t_0; t -= C)
    {
        __syncthreads();
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        w_[i] = __expf(-__expf(float(_w[t])));
        __syncthreads();

        const float gyy = float(_gy[t]);
        float gv = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = sdddd[j];
            float x = gyy * r[j];
            
            gv += (u_[j] * x + s) * k[j];
            s = x + s * w_[j];
        }
        _gv[t] = F_out(gv);
    }
}

template <typename F_in, typename F_out>
__global__ void kernel_backward_201(const int B, const int T, const int C, const int H,
    const F_in *__restrict__ const _r, const F_in *__restrict__ const _k, const F_in *__restrict__ const _v, const F_in *__restrict__ _w, 
    const F_in *__restrict__ _u, const F_out *__restrict__ const _gy,
    F_out *__restrict__ const _gw)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float v[_N_], gy[_N_];
    float saaaa[_N_] = {0}, sbbbb[_T_-2] = {0}, scccc[_N_] = {0};

    const int t_0 = b*T*C + h*_N_ + i;
    const int t_1 = t_0 + C;
    const int t_2 = t_0 + 2*C;
    const int t_T_1 = t_0 + (T-1)*C;

    for (int t = t_T_1; t > t_1; t -= C)
    {
        __syncthreads();
        gy[i] = float(_gy[t]);
        v[i] = float(_v[t-2*C]);
        __syncthreads();

        const float r = float(_r[t]);
        const float w = __expf(-__expf(float(_w[t-C])));
        float sum = 0.0f;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = saaaa[j];
            float x = r * gy[j];
            s = (s + x) * w;
            sum += s * v[j];
        }
        sbbbb[(t-t_2)/C] = sum * float(_k[t-2*C]);
    }

    float sss = sbbbb[0];
    _gw[t_0] = 0;
    _gw[t_1] = F_out(sss * -__expf(float(_w[t_1])));

    for (int t = t_2; t < t_T_1; t += C)
    {
        __syncthreads();
        gy[i] = float(_gy[t]);
        v[i] = float(_v[t-2*C]);
        __syncthreads();

        const float w = __expf(-__expf(float(_w[t-C])));
        const float k = float(_k[t-2*C]);
        float sum = 0.0f;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = scccc[j];
            float x = k * v[j];
            s = (s + x) * w;
            sum += s * gy[j];
        }
        sss += sbbbb[(t-t_1)/C] - (sum * float(_r[t]));
        _gw[t] = F_out(sss * -__expf(float(_w[t])));
    }
    _gw[t_T_1] = 0;
}





template <typename T_in, typename T_out>
void HostApplyRWKVWithState(hipStream_t stream,int B, int T, int C, int H, bool S, const int32_t* state_map,
    const T_in *input_r,const T_in *input_k,const T_in *input_v,
    const T_in *input_w,const T_in *input_u,T_out *input_s, T_out *output_y, T_out *output_s) {
  assert(H*_N_ == C);
  //assert(_N_%4 == 0);
  kernel_forward_state<<<dim3(B * H), dim3(_N_), _N_ * 4 * sizeof(float),stream>>>(B, T, C, H, S, state_map, input_r, input_k, input_v, input_w, input_u,input_s, output_y,output_s);
  
}



template <typename T_in, typename T_out>
void HostApplyRWKV(hipStream_t stream,int B, int T, int C, int H,
    const T_in *input_r,const T_in *input_k,const T_in *input_v,
    const T_in *input_w,const T_in *input_u,T_out *output_y) {
  assert(H*_N_ == C);
  //assert(_N_%4 == 0);
  kernel_forward<<<dim3(B * H), dim3(_N_), _N_ * 4 * sizeof(float),stream>>>(B, T, C, H, input_r, input_k, input_v, input_w, input_u, output_y);
  
}
//todo 为kernel设置正确的sharememory大小
template <typename T_in, typename T_out>
void HostApplyGradient(hipStream_t stream,int B, int T, int C, int H, 
T_in *r, T_in *k, T_in *v, T_in *w, T_in *u, T_out *gy, T_out *gr, T_out *gk, T_out *gv, T_out *gw, T_out *gu)
{
    assert(H*_N_ == C);
    kernel_backward_101<<<dim3(B * H), dim3(_N_),_N_ * 2 * sizeof(float),stream >>>(B, T, C, H, r, k, v, w, u, gy, gr, gu);
    kernel_backward_102<<<dim3(B * H), dim3(_N_),_N_ * 2 * sizeof(float),stream  >>>(B, T, C, H, r, k, v, w, u, gy, gk);
    kernel_backward_103<<<dim3(B * H), dim3(_N_),_N_ * 4 * sizeof(float),stream >>>(B, T, C, H, r, k, v, w, u, gy, gv);
    kernel_backward_201<<<dim3(B * H), dim3(_N_),_N_ * 2 * sizeof(float),stream  >>>(B, T, C, H, r, k, v, w, u, gy, gw);
}

}


namespace gpu_ops {

void rwkv_forward_fn(hipStream_t stream, void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
  const WKVDescriptor &d = *UnpackDescriptor<WKVDescriptor>(opaque, opaque_len);

  DISPATCH_Vector_TYPES(
    d.x_type, d.y_type, "rwkv_forward_kernel",
    HostApplyRWKV<input_type, output_type>(
        stream, d.B, d.T, d.C, d.H, 
        static_cast<input_type *>(buffers[0]),static_cast<input_type *>(buffers[1]),static_cast<input_type *>(buffers[2]),
        static_cast<input_type *>(buffers[3]),static_cast<input_type *>(buffers[4]),static_cast<output_type *>(buffers[5])
    );
  )
}


void rwkv_forward_with_state_fn(hipStream_t stream, void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
  const WKVDescriptor &d = *UnpackDescriptor<WKVDescriptor>(opaque, opaque_len);

  DISPATCH_Vector_TYPES(
    d.x_type, d.y_type, "rwkv_forward_with_state_kernel",
    if(d.S){
      HostApplyRWKVWithState<input_type, output_type>(
          stream, d.B, d.T, d.C, d.H, true, static_cast<int32_t *>(buffers[0])/*map*/,
          static_cast<input_type *>(buffers[1])/*r*/,static_cast<input_type *>(buffers[2])/*k*/,static_cast<input_type *>(buffers[3])/*v*/,
          static_cast<input_type *>(buffers[4])/*w*/,static_cast<input_type *>(buffers[5])/*u*/,static_cast<output_type *>(buffers[6])/*s*/,
          static_cast<output_type *>(buffers[7])/*y*/,static_cast<output_type *>(buffers[8])/*ys*/
      );
    }else{
      HostApplyRWKVWithState<input_type, output_type>(
          stream, d.B, d.T, d.C, d.H, false, nullptr,
          static_cast<input_type *>(buffers[0])/*r*/,static_cast<input_type *>(buffers[1])/*k*/,static_cast<input_type *>(buffers[2])/*v*/,
          static_cast<input_type *>(buffers[3])/*w*/,static_cast<input_type *>(buffers[4])/*u*/,nullptr/*s*/,
          static_cast<output_type *>(buffers[5])/*y*/,static_cast<output_type *>(buffers[6])/*ys*/
      );
    }
  )
}


void rwkv_backward_fn(hipStream_t stream, void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
  const WKVDescriptor &d = *UnpackDescriptor<WKVDescriptor>(opaque, opaque_len);

  DISPATCH_Vector_TYPES(
    d.x_type, d.y_type, "rwkv_backward_kernel",
    HostApplyGradient<input_type, output_type>(
        stream, d.B, d.T, d.C, d.H, 
        static_cast<input_type *>(buffers[0]),static_cast<input_type *>(buffers[1]),static_cast<input_type *>(buffers[2]),
        static_cast<input_type *>(buffers[3]),static_cast<input_type *>(buffers[4]),static_cast<output_type *>(buffers[5]),
        static_cast<output_type *>(buffers[6]),static_cast<output_type *>(buffers[7]),static_cast<output_type *>(buffers[8]),
        static_cast<output_type *>(buffers[9]),static_cast<output_type *>(buffers[10])
    );
  )
}


} // namespace gpu_ops

