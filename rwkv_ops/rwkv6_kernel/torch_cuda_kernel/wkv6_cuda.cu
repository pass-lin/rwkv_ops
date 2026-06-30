#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef float fp32;
typedef at::Half fp16;





template <typename F_in,typename F_out>
__device__ void kernel_forward_core(const int B, const int T, const int C, const int H, const int b, const int h, const int i, const float* state,
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
__global__ void kernel_forward_state(const int B, const int T, const int C, const int H, const bool is_custom_state, const int64_t* state_map,
                               const F_in *__restrict__ const _r, const F_in *__restrict__ const _k, const F_in *__restrict__ const _v, const F_in *__restrict__ _w, const F_in *__restrict__ _u,
                               const F_out *__restrict__ _s, F_out *__restrict__ const _y, F_out *__restrict__ const _ys)
{
  const int b = blockIdx.x / H;
  const int h = blockIdx.x % H;
  const int i = threadIdx.x;
  float state[_N_] = {0}; 

  if(is_custom_state){
    //printf("init\n");  
    assert(state_map[b] >=0 && state_map[b] < B);

    const int64_t input_state_offset = state_map[b] * H * _N_ *_N_ + h * _N_ * _N_ + i;
    
    for(int j= 0; j< _N_; j++){
      state[j] = float(_s[j * _N_ + input_state_offset]);
    }
  }
  
  
  const int64_t current_state_offset = b * H * _N_ *_N_ + h * _N_ * _N_ + i;

  kernel_forward_core(B, T, C, H, b, h, i, state, _r, _k, _v, _w, _u, _y);
  for(int j=0; j< _N_; j++){
    _ys[j * _N_ + current_state_offset] = F_out(state[j]);
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








template<typename F_in,typename F_out>
void cuda_forward(int B, int T, int C, int H, F_in *r, F_in *k, F_in *v, F_in *w, F_in *u, F_out *y)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, y);
}

template<typename F_in,typename F_out>
void cuda_forward_with_state(int B, int T, int C, int H, bool S, int64_t *map, F_in *r, F_in *k, F_in *v, F_in *w, F_in *u, F_out *s, F_out *y, F_out *ys)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    if(S){
        kernel_forward_state<F_in,F_out><<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, S, map, r, k, v, w, u, s, y, ys);
    }else{
        kernel_forward_state<F_in,F_out><<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, S, nullptr, r, k, v, w, u, nullptr, y, ys);
    }
}


template<typename F_in,typename F_out>
void cuda_backward(int B, int T, int C, int H, F_in *r, F_in *k, F_in *v, F_in *w, F_in *u, F_out *gy, F_out *gr, F_out *gk, F_out *gv, F_out *gw, F_out *gu)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_backward_101<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, gy, gr, gu);
    kernel_backward_102<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, gy, gk);
    kernel_backward_103<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, gy, gv);
    kernel_backward_201<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, gy, gw);
}

void cuda_forward_bf16(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, bf16 *w, bf16 *u, bf16 *y){
    cuda_forward<bf16,bf16>(B, T, C, H, r, k, v, w, u, y);
}
void cuda_backward_bf16(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, bf16 *w, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu){
    cuda_backward<bf16,bf16>(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu);
}

void cuda_forward_fp16(int B, int T, int C, int H, fp16 *r, fp16 *k, fp16 *v, fp16 *w, fp16 *u, fp32 *y){
    cuda_forward<fp16,fp32>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward_fp16(int B, int T, int C, int H, fp16 *r, fp16 *k, fp16 *v, fp16 *w, fp16 *u, fp32 *gy, fp32 *gr, fp32 *gk, fp32 *gv, fp32 *gw, fp32 *gu){
    cuda_backward<fp16,fp32>(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu);
}


void cuda_forward_fp32(int B, int T, int C, int H, fp32 *r, fp32 *k, fp32 *v, fp32 *w, fp32 *u, fp32 *y){
     cuda_forward<fp32,fp32>(B, T, C, H, r, k, v, w, u, y);

}
void cuda_backward_fp32(int B, int T, int C, int H, fp32 *r, fp32 *k, fp32 *v, fp32 *w, fp32 *u, fp32 *gy, fp32 *gr, fp32 *gk, fp32 *gv, fp32 *gw, fp32 *gu){
    cuda_backward<fp32,fp32>(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu);
}




void cuda_forward_with_state_bf16(int B, int T, int C, int H, bool S, int64_t *map, bf16 *r, bf16 *k, bf16 *v, bf16 *w, bf16 *u, bf16 *s, bf16 *y, bf16 *ys){
    cuda_forward_with_state<bf16,bf16>(B, T, C, H, S, map, r, k, v, w, u, s, y, ys);
}


void cuda_forward_with_state_fp16(int B, int T, int C, int H, bool S,int64_t *map, fp16 *r, fp16 *k, fp16 *v, fp16 *w, fp16 *u,fp32 *s, fp32 *y, fp32 *ys){
    cuda_forward_with_state<fp16,fp32>(B, T, C, H, S, map, r, k, v, w, u, s, y, ys);
}

void cuda_forward_with_state_fp32(int B, int T, int C, int H, bool S,int64_t *map, fp32 *r, fp32 *k, fp32 *v, fp32 *w, fp32 *u, fp32 *s,  fp32 *y, fp32 *ys){
     cuda_forward_with_state<fp32,fp32>(B, T, C, H, S, map, r, k, v, w, u, s, y, ys);

}