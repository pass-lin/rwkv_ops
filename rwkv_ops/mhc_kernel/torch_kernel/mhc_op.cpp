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

    void cuda_stream_distribute_fwd(nv_bfloat16* out, const nv_bfloat16* inp, const float* H, int64_t B, int64_t T, int n, int64_t C, cudaStream_t stream);
    void cuda_stream_distribute_bwd(nv_bfloat16* d_inp, float* d_H, const nv_bfloat16* grad, const nv_bfloat16* inp, const float* H, int64_t B, int64_t T, int n, int64_t C, cudaStream_t stream);

    void cuda_mhc_post_op_fwd(nv_bfloat16* out, const nv_bfloat16* layer_out, const nv_bfloat16* x_expanded, 
                             const float* H_post, const float* H_res, int64_t B, int64_t T, int n, int64_t C, cudaStream_t stream);

    void cuda_mhc_post_op_bwd(nv_bfloat16* d_layer_out, nv_bfloat16* d_x_expanded, float* d_H_post, float* d_H_res,
                             const nv_bfloat16* grad_next, const nv_bfloat16* layer_out, const nv_bfloat16* x_expanded,
                             const float* H_post, const float* H_res, int64_t B, int64_t T, int n, int64_t C, cudaStream_t stream);

    void cuda_mhc_pre_op_fwd(nv_bfloat16* x_layer_in, float* H_pre, float* H_post, float* H_res,
                            const nv_bfloat16* x_expanded, const float* h_pre_raw, const float* h_post_raw, const float* h_res_raw,
                            int64_t B, int64_t T, int n, int64_t C, int sinkhorn_iters, float eps, cudaStream_t stream);

    void cuda_mhc_pre_op_bwd(nv_bfloat16* d_x_expanded, float* d_h_pre_raw, float* d_h_post_raw, float* d_h_res_raw,
                            const nv_bfloat16* grad_layer_in, const float* grad_H_post, const float* grad_H_res,
                            const nv_bfloat16* x_expanded, const float* H_pre, const float* H_post, const float* H_res_out, const float* H_res_in_raw,
                            int64_t B, int64_t T, int n, int64_t C, int sinkhorn_iters, float eps, cudaStream_t stream);
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
torch::Tensor stream_distribute_fwd(torch::Tensor inp, torch::Tensor H) {
    // inp: [B, T, C], H: [B, T, n]
    int64_t B = inp.size(0);
    int64_t T = inp.size(1);
    int64_t C = inp.size(2);
    int n = H.size(2);

    auto out = torch::empty({B, T, n, C}, inp.options());

    mhc::cuda_stream_distribute_fwd(
        (nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        (nv_bfloat16*)inp.contiguous().data_ptr<at::BFloat16>(),
        H.contiguous().data_ptr<float>(),
        B, T, n, C, 
        at::cuda::getCurrentCUDAStream()
    );
    return out;
}

std::vector<torch::Tensor> stream_distribute_backward(torch::Tensor grad, torch::Tensor inp, torch::Tensor H) {
    int64_t B = inp.size(0);
    int64_t T = inp.size(1);
    int64_t C = inp.size(2);
    int n = H.size(2);

    auto d_inp = torch::empty_like(inp);
    auto d_H = torch::empty_like(H);

    mhc::cuda_stream_distribute_bwd(
        (nv_bfloat16*)d_inp.data_ptr<at::BFloat16>(),
        d_H.data_ptr<float>(),
        (nv_bfloat16*)grad.contiguous().data_ptr<at::BFloat16>(),
        (nv_bfloat16*)inp.contiguous().data_ptr<at::BFloat16>(),
        H.contiguous().data_ptr<float>(),
        B, T, n, C, 
        at::cuda::getCurrentCUDAStream()
    );
    return {d_inp, d_H};
}
torch::Tensor mhc_post_op_forward(torch::Tensor layer_out, torch::Tensor x_expanded, torch::Tensor H_post, torch::Tensor H_res) {
    int64_t B = layer_out.size(0);
    int64_t T = layer_out.size(1);
    int64_t C = layer_out.size(2);
    int n = H_post.size(2);

    auto out = torch::empty_like(x_expanded);
    mhc::cuda_mhc_post_op_fwd(
        (nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        (nv_bfloat16*)layer_out.contiguous().data_ptr<at::BFloat16>(),
        (nv_bfloat16*)x_expanded.contiguous().data_ptr<at::BFloat16>(),
        H_post.contiguous().data_ptr<float>(),
        H_res.contiguous().data_ptr<float>(),
        B, T, n, C, at::cuda::getCurrentCUDAStream()
    );
    return out;
}

// 反向 Torch 接口 (全量融合)
std::vector<torch::Tensor> mhc_post_op_backward(torch::Tensor grad_next, torch::Tensor layer_out, torch::Tensor x_expanded, torch::Tensor H_post, torch::Tensor H_res) {
    int64_t B = layer_out.size(0);
    int64_t T = layer_out.size(1);
    int64_t C = layer_out.size(2);
    int n = H_post.size(2);

    auto d_layer_out = torch::empty_like(layer_out);
    auto d_x_expanded = torch::empty_like(x_expanded);
    // 参数梯度使用 zeros，因为内核内部是原子累加
    auto d_H_post = torch::zeros_like(H_post);
    auto d_H_res = torch::zeros_like(H_res);

    mhc::cuda_mhc_post_op_bwd(
        (nv_bfloat16*)d_layer_out.data_ptr<at::BFloat16>(),
        (nv_bfloat16*)d_x_expanded.data_ptr<at::BFloat16>(),
        d_H_post.data_ptr<float>(),
        d_H_res.data_ptr<float>(),
        (nv_bfloat16*)grad_next.contiguous().data_ptr<at::BFloat16>(),
        (nv_bfloat16*)layer_out.contiguous().data_ptr<at::BFloat16>(),
        (nv_bfloat16*)x_expanded.contiguous().data_ptr<at::BFloat16>(),
        H_post.contiguous().data_ptr<float>(),
        H_res.contiguous().data_ptr<float>(),
        B, T, n, C, at::cuda::getCurrentCUDAStream()
    );

    return {d_layer_out, d_x_expanded, d_H_post, d_H_res};
}
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

// 声明 CUDA 包装函数（定义在 mhc_cuda.cu 中）
namespace mhc {
    void cuda_mhc_pre_op_fwd(nv_bfloat16* x_layer_in, float* H_pre, float* H_post, float* H_res,
                            const nv_bfloat16* x_expanded, const float* h_pre_raw, const float* h_post_raw, const float* h_res_raw,
                            int64_t B, int64_t T, int n, int64_t C, int sinkhorn_iters, float eps, cudaStream_t stream);

    void cuda_mhc_pre_op_bwd(nv_bfloat16* d_x_expanded, float* d_h_pre_raw, float* d_h_post_raw, float* d_h_res_raw,
                            const nv_bfloat16* grad_layer_in, const float* grad_H_post, const float* grad_H_res,
                            const nv_bfloat16* x_expanded, const float* H_pre, const float* H_post, const float* H_res_out, const float* H_res_in_raw,
                            int64_t B, int64_t T, int n, int64_t C, int sinkhorn_iters, float eps, cudaStream_t stream);
}

// ----------------------------------------------------------------------------
// 1. Forward 接口：全部改为 zeros 确保输出纯净
// ----------------------------------------------------------------------------
std::vector<torch::Tensor> mhc_pre_op_forward(
    torch::Tensor x_expanded, torch::Tensor h_pre_raw, torch::Tensor h_post_raw, torch::Tensor h_res_raw,
    int sinkhorn_iters, float eps) 
{
    int64_t B = x_expanded.size(0);
    int64_t T = x_expanded.size(1);
    int n = x_expanded.size(2);
    int64_t C = x_expanded.size(3);

    // 使用 zeros 替代 empty，防止 kernel 未覆盖区域产生脏数据污染 Sinkhorn
    auto x_layer_in = torch::zeros({B, T, C}, x_expanded.options());
    auto H_pre = torch::zeros({B, T, n}, h_pre_raw.options());
    auto H_post = torch::zeros({B, T, n}, h_post_raw.options());
    auto H_res = torch::zeros({B, T, n, n}, h_res_raw.options());

    mhc::cuda_mhc_pre_op_fwd(
        (nv_bfloat16*)x_layer_in.data_ptr<at::BFloat16>(),
        H_pre.data_ptr<float>(),
        H_post.data_ptr<float>(),
        H_res.data_ptr<float>(),
        (nv_bfloat16*)x_expanded.contiguous().data_ptr<at::BFloat16>(),
        h_pre_raw.contiguous().data_ptr<float>(),
        h_post_raw.contiguous().data_ptr<float>(),
        h_res_raw.contiguous().data_ptr<float>(),
        B, T, n, C, sinkhorn_iters, eps, 
        c10::cuda::getCurrentCUDAStream()
    );

    return {x_layer_in, H_pre, H_post, H_res};
}

// ----------------------------------------------------------------------------
// 2. Backward 接口：全部改为 zeros 确保梯度累加安全
// ----------------------------------------------------------------------------
std::vector<torch::Tensor> mhc_pre_op_backward(
    torch::Tensor grad_layer_in, torch::Tensor grad_H_post, torch::Tensor grad_H_res,
    torch::Tensor x_expanded, torch::Tensor H_pre, torch::Tensor H_post, 
    torch::Tensor H_res_out, torch::Tensor h_res_raw,
    int sinkhorn_iters, float eps) 
{
    int64_t B = x_expanded.size(0);
    int64_t T = x_expanded.size(1);
    int n = x_expanded.size(2);
    int64_t C = x_expanded.size(3);

    // 梯度 Tensor 必须清零，因为内核可能涉及原子加或特定线程写回
    auto d_x_expanded = torch::zeros_like(x_expanded);
    auto d_h_pre_raw = torch::zeros_like(H_pre); 
    auto d_h_post_raw = torch::zeros_like(H_post);
    auto d_h_res_raw = torch::zeros({B, T, n * n}, h_res_raw.options());

    mhc::cuda_mhc_pre_op_bwd(
        (nv_bfloat16*)d_x_expanded.data_ptr<at::BFloat16>(),
        d_h_pre_raw.data_ptr<float>(),
        d_h_post_raw.data_ptr<float>(),
        d_h_res_raw.data_ptr<float>(),
        (nv_bfloat16*)grad_layer_in.contiguous().data_ptr<at::BFloat16>(),
        grad_H_post.contiguous().data_ptr<float>(),
        grad_H_res.contiguous().data_ptr<float>(),
        (nv_bfloat16*)x_expanded.contiguous().data_ptr<at::BFloat16>(),
        H_pre.contiguous().data_ptr<float>(),
        H_post.contiguous().data_ptr<float>(),
        H_res_out.contiguous().data_ptr<float>(),
        h_res_raw.contiguous().data_ptr<float>(),
        B, T, n, C, sinkhorn_iters, eps,
        c10::cuda::getCurrentCUDAStream()
    );

    return {d_x_expanded, d_h_pre_raw, d_h_post_raw, d_h_res_raw};
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
    m.def("stream_distribute_fwd", &stream_distribute_fwd, "Stream Distribute Forward");
    m.def("stream_distribute_bwd", &stream_distribute_backward, "Stream Distribute Backward");
    m.def("mhc_post_op_fwd", &mhc_post_op_forward);
    m.def("mhc_post_op_bwd", &mhc_post_op_backward);
    m.def("mhc_pre_op_bwd", &mhc_pre_op_backward);
    m.def("mhc_pre_op_fwd", &mhc_pre_op_forward);
}
