// Copyright (c) 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LIBSPIRV_OPT_SPLIT_COMBINED_IMAGE_SAMPLER_PASS_H_
#define LIBSPIRV_OPT_SPLIT_COMBINED_IMAGE_SAMPLER_PASS_H_

#include <functional>
#include <tuple>

#include "pass.h"

namespace spvtools {
namespace opt {

// Replaces each combined-image sampler variable with an image variable
// and a sampler variable.
//
// Binding numbers are remapped as follows:
// * For a combined image+sampler at binding k, its corresponding
//   image is at binding 2*k, and its corresponding sampler is at 2*k+1
// * For other resources, binding k is remapped to binding 2*k.
//
// This simple scheme wastes numbers, but it should be fine for downstream
// use in WebGPU.
//
// Limitations:
// * Does nothing for now.
// * Does not handle arrays-of-resources
class SplitCombinedImageSamplerPass : public Pass {
 public:
  virtual ~SplitCombinedImageSamplerPass() override = default;
  const char* name() const override { return "split-combined-image-sampler"; }
  Status Process() override;
};
}  // namespace opt
}  // namespace spvtools
#endif  // LIBSPIRV_OPT_SPLIT_COMBINED_IMAGE_SAMPLER_PASS_H_
