#include <torch/extension.h>
#include <cuda_bf16.h>

using bf = __nv_bfloat16;

/* 前向声明：与 CUDA 侧一致 */
void cuda_forward_single_step(
    int B, int H,
    bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
    float* h0, bf* y, float* h1);

/* PyTorch 入口：只负责张量解包与类型转换 */
void forward_single_step(
    torch::Tensor w,   // (B, H, K)  bfloat16
    torch::Tensor q,   // (B, H, K)  bfloat16
    torch::Tensor k,   // (B, H, K)  bfloat16
    torch::Tensor v,   // (B, H, K)  bfloat16
    torch::Tensor a,   // (B, H, K)  bfloat16
    torch::Tensor b,   // (B, H, K)  bfloat16
    torch::Tensor h0,  // (B, H, K, K)  float32
    torch::Tensor y,   // (B, H, K)  bfloat16  输出
    torch::Tensor h1)  // (B, H, K, K)  float32  输出
{
    /* 基本校验 */
    TORCH_CHECK(w.device().is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(w.dtype() == torch::kBFloat16, "w/q/k/v/a/b must be bfloat16");
    TORCH_CHECK(h0.dtype() == torch::kFloat32, "h0/h1 must be float32");
    TORCH_CHECK(w.is_contiguous(), "All tensors must be contiguous");

    const int B = w.size(0);
    const int H = w.size(1);
    const int K = w.size(2);

    cuda_forward_single_step(
        B, H,
        reinterpret_cast<bf*>(w.data_ptr()),
        reinterpret_cast<bf*>(q.data_ptr()),
        reinterpret_cast<bf*>(k.data_ptr()),
        reinterpret_cast<bf*>(v.data_ptr()),
        reinterpret_cast<bf*>(a.data_ptr()),
        reinterpret_cast<bf*>(b.data_ptr()),
        h0.data_ptr<float>(),
        reinterpret_cast<bf*>(y.data_ptr()),
        h1.data_ptr<float>());
}

/* 注册算子 */
TORCH_LIBRARY(wind_backstepping_single_step, m) {
    m.def("forward_single_step("
          "Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, "
          "Tensor h0, Tensor(a!) y, Tensor(b!) h1) -> ()");
}

TORCH_LIBRARY_IMPL(wind_backstepping_single_step, CUDA, m) {
    m.impl("forward_single_step", forward_single_step);
}