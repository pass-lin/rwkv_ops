#include <torch/extension.h>
#include <cuda_bf16.h>

using bf = __nv_bfloat16;

void cuda_forward_sn(int B, int T, int H,
                     bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                     const float* tau, bf* y, float* s, float* sa, float* h0);

void cuda_backward_sn(int B, int T, int H,
                      bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                      const float* tau, bf* dy,
                      float* s, float* sa, float* dht, float* dh0, float* dtau,
                      bf* dw, bf* dq, bf* dk, bf* dv, bf* da, bf* db);

void cuda_forward_inference_sn(int B, int T, int H,
                               bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                               const float* tau, bf* y, float* s, float* h0);

void forward_sn(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v,
                torch::Tensor &a, torch::Tensor &b, torch::Tensor &tau,
                torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa, torch::Tensor &h0) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward_sn(B, T, H,
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(),
        (bf*)a.data_ptr(), (bf*)b.data_ptr(),
        (float*)tau.data_ptr(),
        (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr(), (float*)h0.data_ptr());
}

void backward_sn(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v,
                 torch::Tensor &a, torch::Tensor &b, torch::Tensor &tau, torch::Tensor &dy,
                 torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dht, torch::Tensor &dh0,
                 torch::Tensor &dtau,
                 torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk,
                 torch::Tensor &dv, torch::Tensor &da, torch::Tensor &db) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward_sn(B, T, H,
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(),
        (bf*)a.data_ptr(), (bf*)b.data_ptr(),
        (float*)tau.data_ptr(), (bf*)dy.data_ptr(),
        (float*)s.data_ptr(), (float*)sa.data_ptr(),
        (float*)dht.data_ptr(), (float*)dh0.data_ptr(), (float*)dtau.data_ptr(),
        (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(),
        (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr());
}

void forward_inference_sn(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v,
                          torch::Tensor &a, torch::Tensor &b, torch::Tensor &tau,
                          torch::Tensor &y, torch::Tensor &s, torch::Tensor &h0) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward_inference_sn(B, T, H,
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(),
        (bf*)a.data_ptr(), (bf*)b.data_ptr(),
        (float*)tau.data_ptr(),
        (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)h0.data_ptr());
}

TORCH_LIBRARY(wind_backstepping_sn, m) {
    m.def("forward_sn(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor tau, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa, Tensor(d!) h0) -> ()");
    m.def("backward_sn(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor tau, Tensor dy, Tensor s, Tensor sa, Tensor dht, Tensor(a!) dh0, Tensor(b!) dtau, Tensor(c!) dw, Tensor(d!) dq, Tensor(e!) dk, Tensor(f!) dv, Tensor(g!) da, Tensor(h!) db) -> ()");
    m.def("forward_inference_sn(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor tau, Tensor(a!) y, Tensor(b!) s, Tensor(c!) h0) -> ()");
}

TORCH_LIBRARY_IMPL(wind_backstepping_sn, CUDA, m) {
    m.impl("forward_sn", &forward_sn);
    m.impl("backward_sn", &backward_sn);
    m.impl("forward_inference_sn", &forward_inference_sn);
}
