#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../common_kernel/include/mhc_types.h"
#include "../common_kernel/kernels/sinkhorn_knopp.cuh"
#include "../common_kernel/kernels/rmsnorm.cuh"
#include "../common_kernel/kernels/stream_mix.cuh"
#include "../common_kernel/kernels/stream_aggregate.cuh"
#include "../common_kernel/kernels/stream_distribute.cuh"

namespace mhc {

// --- Sinkhorn 包装 ---
void cuda_sinkhorn_fwd(float* out, const float* inp, int64_t B, int64_t M, int64_t N, int iters, float eps, cudaStream_t stream) {
    for (int64_t b = 0; b < B; b++) {
        mhc::sinkhorn_knopp_forward(out + b * M * N, inp + b * M * N, (int)M, (int)N, iters, eps, stream);
    }
}

void cuda_sinkhorn_bwd(float* d_inp, const float* grad, const float* M_out, const float* M_inp, int64_t B, int64_t N, int iters, float eps, cudaStream_t stream) {
    for (int64_t b = 0; b < B; b++) {
        mhc::sinkhorn_knopp_backward(d_inp + b * N * N, grad + b * N * N, M_out + b * N * N, M_inp + b * N * N, (int)N, iters, eps, stream);
    }
}

// --- RMSNorm 包装 ---
void cuda_rmsnorm_fwd(nv_bfloat16* out, const nv_bfloat16* inp, int64_t N, int64_t C, float eps, cudaStream_t stream) {
    mhc::rmsnorm_forward(reinterpret_cast<mhc::floatX*>(out), reinterpret_cast<const mhc::floatX*>(inp), N, C, eps, stream);
}

void cuda_rmsnorm_bwd(nv_bfloat16* dx, const nv_bfloat16* grad, const nv_bfloat16* x, int64_t N, int64_t C, float eps, cudaStream_t stream) {
    mhc::rmsnorm_backward(reinterpret_cast<mhc::floatX*>(dx), reinterpret_cast<const mhc::floatX*>(grad), reinterpret_cast<const mhc::floatX*>(x), N, C, eps, stream);
}

// --- Stream Mix 包装 ---
void cuda_stream_mix_fwd(nv_bfloat16* out, const nv_bfloat16* inp, const float* M, int64_t B, int64_t T, int n, int64_t C, cudaStream_t stream) {
    mhc::stream_mix_forward(reinterpret_cast<mhc::floatX*>(out), reinterpret_cast<const mhc::floatX*>(inp), M, B, T, n, C, stream);
}

void cuda_stream_mix_bwd(nv_bfloat16* d_inp, float* d_M, const float* grad, const nv_bfloat16* inp, const float* M, int64_t B, int64_t T, int n, int64_t C, cudaStream_t stream) {
    mhc::stream_mix_backward(reinterpret_cast<mhc::floatX*>(d_inp), d_M, grad, reinterpret_cast<const mhc::floatX*>(inp), M, B, T, n, C, stream);
}

// --- 新增：Stream Aggregate 包装 ---
void cuda_stream_aggregate_fwd(nv_bfloat16* out, const nv_bfloat16* inp, const float* H_pre, int64_t B, int64_t T, int n, int64_t C, bool per_token, cudaStream_t stream) {
    mhc::stream_aggregate_forward(reinterpret_cast<mhc::floatX*>(out), reinterpret_cast<const mhc::floatX*>(inp), H_pre, B * T, n, C, per_token, stream);
}

void cuda_stream_aggregate_bwd(nv_bfloat16* d_inp, float* d_H_pre, const float* grad, const nv_bfloat16* inp, const float* H_pre, int64_t B, int64_t T, int n, int64_t C, bool per_token, cudaStream_t stream) {
    mhc::stream_aggregate_backward(reinterpret_cast<mhc::floatX*>(d_inp), d_H_pre, grad, reinterpret_cast<const mhc::floatX*>(inp), H_pre, B * T, n, C, per_token, stream);
}

void cuda_stream_distribute_fwd(
    nv_bfloat16* out, 
    const nv_bfloat16* inp, 
    const float* H, 
    int64_t B, int64_t T, int n, int64_t C, 
    cudaStream_t stream) 
{
    // 计算总线程数（针对单流输入 B*T*C）
    int64_t total_btc = B * T * C;
    dim3 threads(256);
    // x轴覆盖所有的元素，y轴覆盖流索引 n
    dim3 blocks((total_btc + 255) / 256, (unsigned int)n);

    mhc::stream_distribute_fwd_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<mhc::floatX*>(out),
        reinterpret_cast<const mhc::floatX*>(inp),
        H, B, T, n, C
    );
}

void cuda_stream_distribute_bwd(
    nv_bfloat16* d_inp, 
    float* d_H, 
    const nv_bfloat16* grad, 
    const nv_bfloat16* inp, 
    const float* H, 
    int64_t B, int64_t T, int n, int64_t C, 
    cudaStream_t stream) 
{
    // 1. 计算 dx [B, T, C]
    int64_t total_btc = B * T * C;
    dim3 threads_dx(256);
    dim3 blocks_dx((total_btc + 255) / 256);
    
    mhc::stream_distribute_bwd_dx_kernel<<<blocks_dx, threads_dx, 0, stream>>>(
        reinterpret_cast<mhc::floatX*>(d_inp),
        reinterpret_cast<const mhc::floatX*>(grad),
        H, B, T, n, C
    );

    // 2. 计算 dH [B, T, n]
    // 每个 (bt, i) 分配一个 block 进行通道 C 维度的规约
    dim3 threads_dh(256);
    dim3 blocks_dh((unsigned int)(B * T), (unsigned int)n);

    mhc::stream_distribute_bwd_dh_kernel<256><<<blocks_dh, threads_dh, 0, stream>>>(
        d_H,
        reinterpret_cast<const mhc::floatX*>(grad),
        reinterpret_cast<const mhc::floatX*>(inp),
        B, T, n, C
    );
}

} // namespace mhc