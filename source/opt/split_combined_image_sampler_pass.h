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

#include <unordered_map>
#include <vector>

#include "source/diagnostic.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/pass.h"

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

 private:
  // Records failure for the current module, and returns a stream
  // that can be used to provide user error information to the message
  // consumer.
  spvtools::DiagnosticStream Fail();

  // Populates obj_ and ordered_objs_;
  void FindCombinedTextureSamplers();

  struct RemapInfo {
    uint32_t mem_obj_decl = 0;  // the var or parameter.
    uint32_t sampled_image_type = 0;
#if 0
    uint32_t image_type = 0;
    uint32_t sampler_type = 0;
    uint32_t descriptor_set = 0;
    uint32_t original_set = 0;
    uint32_t original_binding = 0;
#endif
  };

  // Cached from the IRContext. Valid while Process() is running.
  analysis::DefUseManager* def_use_mgr_ = nullptr;

  // Maps the ID of a memory object declaration for a combined texture+sampler
  // to remapping information about that object.
  std::unordered_map<uint32_t, RemapInfo> remap_info_;
  // The key of objs_ in the order they were added.
  std::vector<uint32_t> ordered_objs_;

  struct {
    bool failed = false;
  } module_status_;
};
}  // namespace opt
}  // namespace spvtools
#endif  // LIBSPIRV_OPT_SPLIT_COMBINED_IMAGE_SAMPLER_PASS_H_
