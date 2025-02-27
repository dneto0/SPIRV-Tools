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
#include "source/opt/type_manager.h"

namespace spvtools {
namespace opt {

// Replaces each combined-image sampler variable with an image variable
// and a sampler variable.
//
// First cut: use the same binding number. Vulkan allows this, surprisingly.
//
// Second cut: remap the bindings.
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
// * Does not handle calls to functions
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

  spv_result_t EnsureSamplerTypeAppearsFirst();
  spv_result_t RemapVars();
  spv_result_t RemapVar(Instruction* var);
  // Removes instructions queued up for removal during earlier processing
  // stages.
  spv_result_t RemoveDeadInstructions();

  struct RemapInfo {
    uint32_t var_id = 0;
    uint32_t sampled_image_type = 0;
    uint32_t image_type = 0;
  };

  // Cached from the IRContext. Valid while Process() is running.
  analysis::DefUseManager* def_use_mgr_ = nullptr;
  // Cached from the IRContext. Valid while Process() is running.
  analysis::TypeManager* type_mgr_ = nullptr;

  // An OpTypeSampler instruction, if one existed already, or if we created one.
  Instruction* sampler_type_ = nullptr;
  // A pointer-to-sampler instruction, if one existed already, or if we created
  // one.
  Instruction* ptr_sampler_type_ = nullptr;
  // A pointer-to-sampled-image-type instruction, if one existed already.
  Instruction* ptr_sampled_image_type_ = nullptr;

  // Maps the ID of a memory object declaration for a combined texture+sampler
  // to remapping information about that object.
  std::unordered_map<uint32_t, RemapInfo> remap_info_;
  // The instructions added to remap_info_, in the order they were added.
  std::vector<Instruction*> ordered_objs_;

  // The instructions to be removed.
  std::vector<Instruction*> dead_;
};
}  // namespace opt
}  // namespace spvtools
#endif  // LIBSPIRV_OPT_SPLIT_COMBINED_IMAGE_SAMPLER_PASS_H_
