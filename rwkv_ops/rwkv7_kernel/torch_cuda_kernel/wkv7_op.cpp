#include <torch/extension.h>
#include <cuda_bf16.h>

using bf = __nv_bfloat16;

/* ----------- 原版函数声明（保持不变）----------- */
void cuda_forward(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* z, bf* a, bf* y, float* s, float* sa, float* h0);
void cuda_backward(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* z, bf* a, bf* dy,
    float* s, float* sa, float* dht, float* dh0, bf* dw, bf* dq, bf* dk, bf* dv, bf* dz, bf* da);
void cuda_forward_inference(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b, bf* y, float* s, float* h0);

/* ----------- 带 Mask 版本函数声明（新增）----------- */
void cuda_forward_with_mask(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                            bf* y, float* s, float* sa, float* h0, float* mask);
void cuda_backward_with_mask(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                             float* mask, bf* dy, float* s, float* sa, float* dht, float* dh0,
                             bf* dw, bf* dq, bf* dk, bf* dv, bf* da, bf* db);
void cuda_forward_inference_with_mask(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b, 
                                      bf* y, float* s, float* h0, float* mask);

/* ----------- 原版 Wrapper（保持不变）----------- */
void forward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, 
             torch::Tensor &z, torch::Tensor &a, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa, torch::Tensor &h0) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, 
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), 
        (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)y.data_ptr(), 
        (float*)s.data_ptr(), (float*)sa.data_ptr(), (float*)h0.data_ptr());
}

void backward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, 
              torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy,
              torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dht, torch::Tensor &dh0,
              torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, 
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), 
        (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)dy.data_ptr(),
        (float*)s.data_ptr(), (float*)sa.data_ptr(), (float*)dht.data_ptr(), (float*)dh0.data_ptr(),
        (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), 
        (bf*)dv.data_ptr(), (bf*)dz.data_ptr(), (bf*)da.data_ptr());
}

void forward_inference(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, 
                      torch::Tensor &a, torch::Tensor &b, torch::Tensor &y, torch::Tensor &s, torch::Tensor &h0) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward_inference(B, T, H,
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), 
        (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)y.data_ptr(), 
        (float*)s.data_ptr(), (float*)h0.data_ptr());
}

/* ----------- 带 Mask Wrapper（新增）----------- */
void forward_with_mask(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, 
                       torch::Tensor &a, torch::Tensor &b, torch::Tensor &mask,
                       torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa, torch::Tensor &h0) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward_with_mask(B, T, H, 
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), 
        (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)y.data_ptr(), 
        (float*)s.data_ptr(), (float*)sa.data_ptr(), (float*)h0.data_ptr(),
        (float*)mask.data_ptr());
}

void backward_with_mask(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, 
                        torch::Tensor &a, torch::Tensor &b, torch::Tensor &mask,
                        torch::Tensor &dy, torch::Tensor &s, torch::Tensor &sa, 
                        torch::Tensor &dht, torch::Tensor &dh0,
                        torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, 
                        torch::Tensor &dv, torch::Tensor &da, torch::Tensor &db) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward_with_mask(B, T, H, 
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), 
        (bf*)a.data_ptr(), (bf*)b.data_ptr(),
        (float*)mask.data_ptr(),
        (bf*)dy.data_ptr(),
        (float*)s.data_ptr(), (float*)sa.data_ptr(), (float*)dht.data_ptr(), (float*)dh0.data_ptr(),
        (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), 
        (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr());
}

void forward_inference_with_mask(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, 
                                torch::Tensor &a, torch::Tensor &b, torch::Tensor &mask,
                                torch::Tensor &y, torch::Tensor &s, torch::Tensor &h0) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward_inference_with_mask(B, T, H,
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), 
        (bf*)a.data_ptr(), (bf*)b.data_ptr(),
        (bf*)y.data_ptr(), 
        (float*)s.data_ptr(), (float*)h0.data_ptr(),
        (float*)mask.data_ptr());
}

/* ----------- 算子注册 ----------- */
TORCH_LIBRARY(wind_backstepping, m) {
    // 原版
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa, Tensor(d!) h0) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor sa, Tensor dht, Tensor(a!) dh0, Tensor(b!) dw, Tensor(c!) dq, Tensor(d!) dk, Tensor(e!) dv, Tensor(f!) dz, Tensor(g!) da) -> ()");
    m.def("forward_inference(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor(b!) s, Tensor(c!) h0) -> ()");
    
    // 带 Mask 版本
    m.def("forward_with_mask(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor mask, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa, Tensor(d!) h0) -> ()");
    m.def("backward_with_mask(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor mask, Tensor dy, Tensor s, Tensor sa, Tensor dht, Tensor(a!) dh0, Tensor(b!) dw, Tensor(c!) dq, Tensor(d!) dk, Tensor(e!) dv, Tensor(f!) da, Tensor(g!) db) -> ()");
    m.def("forward_inference_with_mask(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor mask, Tensor(a!) y, Tensor(b!) s, Tensor(c!) h0) -> ()");
}

TORCH_LIBRARY_IMPL(wind_backstepping, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
    m.impl("forward_inference", &forward_inference);
    
    m.impl("forward_with_mask", &forward_with_mask);
    m.impl("backward_with_mask", &backward_with_mask);
    m.impl("forward_inference_with_mask", &forward_inference_with_mask);
}