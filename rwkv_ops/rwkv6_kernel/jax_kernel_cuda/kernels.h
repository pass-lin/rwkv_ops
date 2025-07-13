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

#ifndef _GPU_OPS_KERNELS_H_
#define _GPU_OPS_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#ifndef _N_
  #define _N_ 8
#endif
#ifndef _T_
  #define _T_ 16
#endif
namespace gpu_ops {

enum ElementType { BF16, F16, F32 };

struct WKVDescriptor {
  int B;
  int T;
  int C;
  int H;
  bool S;
  ElementType x_type;
  ElementType y_type;
};

void rwkv_forward_fn(cudaStream_t stream, void **buffers,
                      const char *opaque,
                      std::size_t opaque_len);
void rwkv_backward_fn(cudaStream_t stream, void **buffers,
                      const char *opaque,
                      std::size_t opaque_len);

void rwkv_forward_with_state_fn(cudaStream_t stream, void **buffers,
                      const char *opaque,
                      std::size_t opaque_len);
} // namespace gpu_ops

#endif
