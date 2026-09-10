#include <cuda_bf16.h>
#include <torch/extension.h>

using bf = __nv_bfloat16;

void cuda_forward_single_step_sane(int B, int H, bf *w, bf *q, bf *k, bf *v,
                                   bf *a, bf *b, const float *tau,
                                   const int8_t *do_sane, float *h0, bf *y,
                                   float *h1);

void forward_single_step_sane(torch::Tensor w, torch::Tensor q, torch::Tensor k,
                              torch::Tensor v, torch::Tensor a, torch::Tensor b,
                              torch::Tensor tau, torch::Tensor do_sane,
                              torch::Tensor h0, torch::Tensor y,
                              torch::Tensor h1) {
  TORCH_CHECK(w.device().is_cuda(), "All tensors must be CUDA");
  TORCH_CHECK(w.dtype() == torch::kBFloat16, "w/q/k/v/a/b must be bfloat16");
  TORCH_CHECK(h0.dtype() == torch::kFloat32, "h0/h1 must be float32");
  TORCH_CHECK(tau.dtype() == torch::kFloat32, "tau must be float32");
  TORCH_CHECK(do_sane.dtype() == torch::kInt8, "do_sane must be int8");
  TORCH_CHECK(w.is_contiguous(), "All tensors must be contiguous");

  const int B = w.size(0);
  const int H = w.size(1);
  const int K = w.size(2);

  cuda_forward_single_step_sane(
      B, H, reinterpret_cast<bf *>(w.data_ptr()),
      reinterpret_cast<bf *>(q.data_ptr()),
      reinterpret_cast<bf *>(k.data_ptr()),
      reinterpret_cast<bf *>(v.data_ptr()),
      reinterpret_cast<bf *>(a.data_ptr()),
      reinterpret_cast<bf *>(b.data_ptr()), tau.data_ptr<float>(),
      do_sane.data_ptr<int8_t>(), h0.data_ptr<float>(),
      reinterpret_cast<bf *>(y.data_ptr()), h1.data_ptr<float>());
}

TORCH_LIBRARY(wind_backstepping_sane_single_step, m) {
  m.def(
      "forward_single_step_sane("
      "Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, "
      "Tensor tau, Tensor do_sane, Tensor h0, Tensor(a!) y, Tensor(b!) h1) -> "
      "()");
}

TORCH_LIBRARY_IMPL(wind_backstepping_sane_single_step, CUDA, m) {
  m.impl("forward_single_step_sane", forward_single_step_sane);
}
