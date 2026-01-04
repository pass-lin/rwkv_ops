#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "../common_kernel/include/mhc_types.h"

namespace mhc {
    // Sinkhorn 接口
    void cuda_sinkhorn_fwd(float* out, const float* inp, int64_t B, int64_t M, int64_t N, int iters, float eps, cudaStream_t stream);
    void cuda_sinkhorn_bwd(float* d_inp, const float* grad, const float* M_out, const float* M_inp, int64_t B, int64_t N, int iters, float eps, cudaStream_t stream);
    
    // RMSNorm 接口
    void cuda_rmsnorm_fwd(nv_bfloat16* out, const nv_bfloat16* inp, int64_t N, int64_t C, float eps, cudaStream_t stream);
    void cuda_rmsnorm_bwd(nv_bfloat16* dx, const nv_bfloat16* grad, const nv_bfloat16* x, int64_t N, int64_t C, float eps, cudaStream_t stream);

    // Stream Mix 接口
    void cuda_stream_mix_fwd(nv_bfloat16* out, const nv_bfloat16* inp, const float* M, int64_t B, int64_t T, int n, int64_t C, cudaStream_t stream);
    void cuda_stream_mix_bwd(nv_bfloat16* d_inp, float* d_M, const float* grad, const nv_bfloat16* inp, const float* M, int64_t B, int64_t T, int n, int64_t C, cudaStream_t stream);

    // 新增：Stream Aggregate 接口
   void cuda_stream_aggregate_fwd(nv_bfloat16* out, const nv_bfloat16* inp, const float* H_pre, int64_t B, int64_t T, int n, int64_t C, bool per_token, cudaStream_t stream);
    void cuda_stream_aggregate_bwd(nv_bfloat16* d_inp, float* d_H_pre, const float* grad, const nv_bfloat16* inp, const float* H_pre, int64_t B, int64_t T, int n, int64_t C, bool per_token, cudaStream_t stream);
}

// --- Sinkhorn 绑定 ---
torch::Tensor sinkhorn_forward(torch::Tensor inp, int iters, float eps) {
    auto out = torch::empty_like(inp);
    int64_t B = inp.numel() / (inp.size(-1) * inp.size(-2));
    mhc::cuda_sinkhorn_fwd(out.data_ptr<float>(), inp.contiguous().data_ptr<float>(), B, inp.size(-2), inp.size(-1), iters, eps, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor sinkhorn_backward(torch::Tensor grad, torch::Tensor out, torch::Tensor inp, int iters, float eps) {
    auto d_inp = torch::empty_like(grad);
    int64_t B = grad.numel() / (grad.size(-1) * grad.size(-1));
    mhc::cuda_sinkhorn_bwd(d_inp.data_ptr<float>(), grad.contiguous().data_ptr<float>(), out.contiguous().data_ptr<float>(), inp.contiguous().data_ptr<float>(), B, grad.size(-1), iters, eps, at::cuda::getCurrentCUDAStream());
    return d_inp;
}

// --- RMSNorm 绑定 ---
torch::Tensor rmsnorm_forward(torch::Tensor inp, float eps) {
    auto out = torch::empty_like(inp);
    int64_t C = inp.size(-1);
    int64_t N = inp.numel() / C;
    mhc::cuda_rmsnorm_fwd((nv_bfloat16*)out.data_ptr<at::BFloat16>(), (nv_bfloat16*)inp.contiguous().data_ptr<at::BFloat16>(), N, C, eps, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor rmsnorm_backward(torch::Tensor grad, torch::Tensor x, float eps) {
    auto dx = torch::empty_like(x);
    int64_t C = x.size(-1);
    int64_t N = x.numel() / C;
    mhc::cuda_rmsnorm_bwd((nv_bfloat16*)dx.data_ptr<at::BFloat16>(), (nv_bfloat16*)grad.contiguous().data_ptr<at::BFloat16>(), (nv_bfloat16*)x.contiguous().data_ptr<at::BFloat16>(), N, C, eps, at::cuda::getCurrentCUDAStream());
    return dx;
}

// --- Stream Mix 绑定 ---
torch::Tensor stream_mix_fwd(torch::Tensor inp, torch::Tensor M) {
    auto B = inp.size(0); auto T = inp.size(1); auto n = inp.size(2); auto C = inp.size(3);
    auto out = torch::empty_like(inp);
    mhc::cuda_stream_mix_fwd((nv_bfloat16*)out.data_ptr<at::BFloat16>(), (nv_bfloat16*)inp.contiguous().data_ptr<at::BFloat16>(), M.contiguous().data_ptr<float>(), B, T, n, C, at::cuda::getCurrentCUDAStream());
    return out;
}

std::vector<torch::Tensor> stream_mix_backward(torch::Tensor grad, torch::Tensor inp, torch::Tensor M) {
    int64_t B = inp.size(0); int64_t T = inp.size(1); int n = inp.size(2); int64_t C = inp.size(3);
    auto d_inp = torch::empty_like(inp);
    auto d_M = torch::empty_like(M);
    mhc::cuda_stream_mix_bwd((nv_bfloat16*)d_inp.data_ptr<at::BFloat16>(), d_M.data_ptr<float>(), grad.contiguous().data_ptr<float>(), (nv_bfloat16*)inp.contiguous().data_ptr<at::BFloat16>(), M.contiguous().data_ptr<float>(), B, T, n, C, at::cuda::getCurrentCUDAStream());
    return {d_inp, d_M};
}

// --- 新增：Stream Aggregate 绑定 ---
torch::Tensor stream_aggregate_fwd(torch::Tensor inp, torch::Tensor H_pre, bool per_token) {
    int64_t B = inp.size(0); int64_t T = inp.size(1); int n = inp.size(2); int64_t C = inp.size(3);
    auto out = torch::empty({B, T, C}, inp.options());
    mhc::cuda_stream_aggregate_fwd((nv_bfloat16*)out.data_ptr<at::BFloat16>(), (nv_bfloat16*)inp.contiguous().data_ptr<at::BFloat16>(), H_pre.contiguous().data_ptr<float>(), B, T, n, C, per_token, at::cuda::getCurrentCUDAStream());
    return out;
}

std::vector<torch::Tensor> stream_aggregate_bwd(torch::Tensor grad, torch::Tensor inp, torch::Tensor H_pre, bool per_token) {
    int64_t B = inp.size(0); int64_t T = inp.size(1); int n = inp.size(2); int64_t C = inp.size(3);
    auto d_inp = torch::empty_like(inp);
    auto d_H_pre = torch::empty_like(H_pre);
    mhc::cuda_stream_aggregate_bwd((nv_bfloat16*)d_inp.data_ptr<at::BFloat16>(), d_H_pre.data_ptr<float>(), grad.contiguous().data_ptr<float>(), (nv_bfloat16*)inp.contiguous().data_ptr<at::BFloat16>(), H_pre.contiguous().data_ptr<float>(), B, T, n, C, per_token, at::cuda::getCurrentCUDAStream());
    return {d_inp, d_H_pre};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sinkhorn_fwd", &sinkhorn_forward);
    m.def("sinkhorn_bwd", &sinkhorn_backward);
    m.def("rmsnorm_fwd", &rmsnorm_forward);
    m.def("rmsnorm_bwd", &rmsnorm_backward);
    m.def("stream_mix_fwd", &stream_mix_fwd);
    m.def("stream_mix_backward", &stream_mix_backward);
    m.def("stream_aggregate_fwd", &stream_aggregate_fwd);
    m.def("stream_aggregate_bwd", &stream_aggregate_bwd);
}