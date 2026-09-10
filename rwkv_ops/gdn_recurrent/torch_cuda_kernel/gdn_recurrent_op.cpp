// PyTorch C++ 绑定：将 Gated DeltaNet recurrent CUDA kernel 包装为 torch.ops。

#include <cuda_bf16.h>
#include <torch/extension.h>

using bf = __nv_bfloat16;

// CUDA 函数前向声明（bfloat16 与 float32 两个显式实例化版本）。
template <typename ET>
void cuda_gdn_forward(int B, int T, int H, float scale, const ET *q,
                      const ET *k, const ET *v, const float *g,
                      const float *beta, const float *h0, ET *o, float *kv_mem,
                      float *chkp, float *inv_q, float *inv_k, float *ht);
template <typename ET>
void cuda_gdn_backward(int B, int T, int H, float scale, const ET *q,
                       const ET *k, const ET *v, const float *g,
                       const float *beta, const ET *dout, const float *dht,
                       const float *kv_mem, const float *inv_q,
                       const float *inv_k, const float *h0, const float *chkp,
                       float *dq, float *dk, float *dv, float *dg, float *dbeta,
                       float *dh0);
template <typename ET>
void cuda_gdn_forward_inference(int B, int T, int H, float scale, const ET *q,
                                const ET *k, const ET *v, const float *g,
                                const float *beta, const float *h0, ET *o,
                                float *ht);
template <typename ET>
void cuda_gdn_single_step(int B, int H, float scale, const ET *q, const ET *k,
                          const ET *v, const float *g, const float *beta,
                          const float *h0, ET *o, float *ht);

extern template void cuda_gdn_forward<bf>(int, int, int, float, const bf *,
                                          const bf *, const bf *, const float *,
                                          const float *, const float *, bf *,
                                          float *, float *, float *, float *,
                                          float *);
extern template void cuda_gdn_forward<float>(int, int, int, float,
                                             const float *, const float *,
                                             const float *, const float *,
                                             const float *, const float *,
                                             float *, float *, float *, float *,
                                             float *, float *);
extern template void
cuda_gdn_backward<bf>(int, int, int, float, const bf *, const bf *, const bf *,
                      const float *, const float *, const bf *, const float *,
                      const float *, const float *, const float *,
                      const float *, const float *, float *, float *, float *,
                      float *, float *, float *);
extern template void cuda_gdn_backward<float>(
    int, int, int, float, const float *, const float *, const float *,
    const float *, const float *, const float *, const float *, const float *,
    const float *, const float *, const float *, const float *, float *,
    float *, float *, float *, float *, float *);
extern template void
cuda_gdn_forward_inference<bf>(int, int, int, float, const bf *, const bf *,
                               const bf *, const float *, const float *,
                               const float *, bf *, float *);
extern template void cuda_gdn_forward_inference<float>(
    int, int, int, float, const float *, const float *, const float *,
    const float *, const float *, const float *, float *, float *);
extern template void cuda_gdn_single_step<bf>(int, int, float, const bf *,
                                              const bf *, const bf *,
                                              const float *, const float *,
                                              const float *, bf *, float *);
extern template void cuda_gdn_single_step<float>(int, int, float, const float *,
                                                 const float *, const float *,
                                                 const float *, const float *,
                                                 const float *, float *,
                                                 float *);

// PyTorch wrapper。输入 layout 均为 head-first：q/k 为 [B, H, T, K]，
// v/o 为 [B, H, T, V]，g/beta 为 [B, H, T]，state 为 [B, H, K, V]。
void forward(torch::Tensor &q, torch::Tensor &k, torch::Tensor &v,
             torch::Tensor &g, torch::Tensor &beta, torch::Tensor &h0,
             double scale, torch::Tensor &o, torch::Tensor &kv_mem,
             torch::Tensor &chkp, torch::Tensor &inv_q, torch::Tensor &inv_k,
             torch::Tensor &ht) {
  int B = q.sizes()[0], H = q.sizes()[1], T = q.sizes()[2];
  if (q.scalar_type() == at::kBFloat16) {
    cuda_gdn_forward<bf>(B, T, H, (float)scale, (const bf *)q.data_ptr(),
                         (const bf *)k.data_ptr(), (const bf *)v.data_ptr(),
                         (const float *)g.data_ptr(),
                         (const float *)beta.data_ptr(),
                         (const float *)h0.data_ptr(), (bf *)o.data_ptr(),
                         (float *)kv_mem.data_ptr(), (float *)chkp.data_ptr(),
                         (float *)inv_q.data_ptr(), (float *)inv_k.data_ptr(),
                         (float *)ht.data_ptr());
  } else if (q.scalar_type() == at::kFloat) {
    cuda_gdn_forward<float>(
        B, T, H, (float)scale, (const float *)q.data_ptr(),
        (const float *)k.data_ptr(), (const float *)v.data_ptr(),
        (const float *)g.data_ptr(), (const float *)beta.data_ptr(),
        (const float *)h0.data_ptr(), (float *)o.data_ptr(),
        (float *)kv_mem.data_ptr(), (float *)chkp.data_ptr(),
        (float *)inv_q.data_ptr(), (float *)inv_k.data_ptr(),
        (float *)ht.data_ptr());
  } else {
    TORCH_CHECK(false, "gdn_recurrent cuda kernel only supports bfloat16 "
                       "or float32 inputs");
  }
}

void backward(torch::Tensor &q, torch::Tensor &k, torch::Tensor &v,
              torch::Tensor &g, torch::Tensor &beta, torch::Tensor &dout,
              torch::Tensor &dht, torch::Tensor &kv_mem, torch::Tensor &inv_q,
              torch::Tensor &inv_k, torch::Tensor &h0, torch::Tensor &chkp,
              double scale, torch::Tensor &dq, torch::Tensor &dk,
              torch::Tensor &dv, torch::Tensor &dg, torch::Tensor &dbeta,
              torch::Tensor &dh0) {
  int B = q.sizes()[0], H = q.sizes()[1], T = q.sizes()[2];
  if (q.scalar_type() == at::kBFloat16) {
    cuda_gdn_backward<bf>(
        B, T, H, (float)scale, (const bf *)q.data_ptr(),
        (const bf *)k.data_ptr(), (const bf *)v.data_ptr(),
        (const float *)g.data_ptr(), (const float *)beta.data_ptr(),
        (const bf *)dout.data_ptr(), (const float *)dht.data_ptr(),
        (const float *)kv_mem.data_ptr(), (const float *)inv_q.data_ptr(),
        (const float *)inv_k.data_ptr(), (const float *)h0.data_ptr(),
        (const float *)chkp.data_ptr(), (float *)dq.data_ptr(),
        (float *)dk.data_ptr(), (float *)dv.data_ptr(), (float *)dg.data_ptr(),
        (float *)dbeta.data_ptr(), (float *)dh0.data_ptr());
  } else if (q.scalar_type() == at::kFloat) {
    cuda_gdn_backward<float>(
        B, T, H, (float)scale, (const float *)q.data_ptr(),
        (const float *)k.data_ptr(), (const float *)v.data_ptr(),
        (const float *)g.data_ptr(), (const float *)beta.data_ptr(),
        (const float *)dout.data_ptr(), (const float *)dht.data_ptr(),
        (const float *)kv_mem.data_ptr(), (const float *)inv_q.data_ptr(),
        (const float *)inv_k.data_ptr(), (const float *)h0.data_ptr(),
        (const float *)chkp.data_ptr(), (float *)dq.data_ptr(),
        (float *)dk.data_ptr(), (float *)dv.data_ptr(), (float *)dg.data_ptr(),
        (float *)dbeta.data_ptr(), (float *)dh0.data_ptr());
  } else {
    TORCH_CHECK(false, "gdn_recurrent cuda kernel only supports bfloat16 "
                       "or float32 inputs");
  }
}

void forward_inference(torch::Tensor &q, torch::Tensor &k, torch::Tensor &v,
                       torch::Tensor &g, torch::Tensor &beta, torch::Tensor &h0,
                       double scale, torch::Tensor &o, torch::Tensor &ht) {
  int B = q.sizes()[0], H = q.sizes()[1], T = q.sizes()[2];
  if (q.scalar_type() == at::kBFloat16) {
    cuda_gdn_forward_inference<bf>(
        B, T, H, (float)scale, (const bf *)q.data_ptr(),
        (const bf *)k.data_ptr(), (const bf *)v.data_ptr(),
        (const float *)g.data_ptr(), (const float *)beta.data_ptr(),
        (const float *)h0.data_ptr(), (bf *)o.data_ptr(),
        (float *)ht.data_ptr());
  } else if (q.scalar_type() == at::kFloat) {
    cuda_gdn_forward_inference<float>(
        B, T, H, (float)scale, (const float *)q.data_ptr(),
        (const float *)k.data_ptr(), (const float *)v.data_ptr(),
        (const float *)g.data_ptr(), (const float *)beta.data_ptr(),
        (const float *)h0.data_ptr(), (float *)o.data_ptr(),
        (float *)ht.data_ptr());
  } else {
    TORCH_CHECK(false, "gdn_recurrent cuda kernel only supports bfloat16 "
                       "or float32 inputs");
  }
}

void single_step(torch::Tensor &q, torch::Tensor &k, torch::Tensor &v,
                 torch::Tensor &g, torch::Tensor &beta, torch::Tensor &h0,
                 double scale, torch::Tensor &o, torch::Tensor &ht) {
  int B = q.sizes()[0], H = q.sizes()[1];
  if (q.scalar_type() == at::kBFloat16) {
    cuda_gdn_single_step<bf>(
        B, H, (float)scale, (const bf *)q.data_ptr(), (const bf *)k.data_ptr(),
        (const bf *)v.data_ptr(), (const float *)g.data_ptr(),
        (const float *)beta.data_ptr(), (const float *)h0.data_ptr(),
        (bf *)o.data_ptr(), (float *)ht.data_ptr());
  } else if (q.scalar_type() == at::kFloat) {
    cuda_gdn_single_step<float>(
        B, H, (float)scale, (const float *)q.data_ptr(),
        (const float *)k.data_ptr(), (const float *)v.data_ptr(),
        (const float *)g.data_ptr(), (const float *)beta.data_ptr(),
        (const float *)h0.data_ptr(), (float *)o.data_ptr(),
        (float *)ht.data_ptr());
  } else {
    TORCH_CHECK(false, "gdn_recurrent cuda kernel only supports bfloat16 "
                       "or float32 inputs");
  }
}

// 算子注册。命名空间由编译宏 TORCH_LIBRARY_NAME 决定，Python 侧按
// (K, V, chunk_size) 传入不同名称实现同一进程内的多版本隔离。
#ifndef TORCH_LIBRARY_NAME
#define TORCH_LIBRARY_NAME gdn_recurrent_cuda
#endif
TORCH_LIBRARY(TORCH_LIBRARY_NAME, m) {
  m.def("forward(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
        "Tensor h0, float scale, Tensor(a!) o, Tensor(b!) kv_mem, "
        "Tensor(c!) chkp, Tensor(d!) inv_q, Tensor(e!) inv_k, Tensor(f!) ht) "
        "-> ()");
  m.def("backward(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
        "Tensor dout, Tensor dht, Tensor kv_mem, Tensor inv_q, Tensor inv_k, "
        "Tensor h0, Tensor chkp, float scale, Tensor(a!) dq, Tensor(b!) dk, "
        "Tensor(c!) dv, Tensor(d!) dg, Tensor(e!) dbeta, Tensor(f!) dh0) "
        "-> ()");
  m.def("forward_inference(Tensor q, Tensor k, Tensor v, Tensor g, "
        "Tensor beta, Tensor h0, float scale, Tensor(a!) o, Tensor(b!) ht) "
        "-> ()");
  m.def("single_step(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
        "Tensor h0, float scale, Tensor(a!) o, Tensor(b!) ht) -> ()");
}

TORCH_LIBRARY_IMPL(TORCH_LIBRARY_NAME, CUDA, m) {
  m.impl("forward", &forward);
  m.impl("backward", &backward);
  m.impl("forward_inference", &forward_inference);
  m.impl("single_step", &single_step);
}
