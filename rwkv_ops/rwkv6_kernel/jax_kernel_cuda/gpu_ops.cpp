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

#include "kernels.h"
#include "pybind11_kernel_helpers.h"

namespace {
pybind11::dict WKVRegistrations() {
  pybind11::dict dict;
  dict["wkv_forward"] =
      gpu_ops::EncapsulateFunction(gpu_ops::rwkv_forward_fn);
  dict["wkv_backward"] =
      gpu_ops::EncapsulateFunction(gpu_ops::rwkv_backward_fn);
  dict["wkv_forward_with_state"] = 
      gpu_ops::EncapsulateFunction(gpu_ops::rwkv_forward_with_state_fn);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("get_rwkv_registrations", &WKVRegistrations);
  m.def("create_rwkv_descriptor",
        [](int B, int T,int C, int H,bool S, gpu_ops::ElementType input_type,gpu_ops::ElementType output_type) {
          return gpu_ops::PackDescriptor(gpu_ops::WKVDescriptor{B, T, C, H, S, input_type, output_type});
        });

  pybind11::enum_<gpu_ops::ElementType>(m, "ElementType")
      .value("BF16", gpu_ops::ElementType::BF16)
      .value("F16", gpu_ops::ElementType::F16)
      .value("F32", gpu_ops::ElementType::F32);

}
} // namespace
