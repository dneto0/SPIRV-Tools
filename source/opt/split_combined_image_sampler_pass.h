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
#include <utility>
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

  void FindCombinedTextureSamplers();

  // Returns the sampler type. If it does not yet exist, then it is created
  // and placed before the first sampled image type.
  Instruction* GetSamplerType();

  // Remaps function types and function declarations.  Each
  // pointer-to-sampled-image-type operand is replaced with a pair of
  // pointer-to-image-type and pointer-to-sampler-type pair.
  spv_result_t RemapFunctions();
  spv_result_t RemapVars();
  spv_result_t RemapVar(Instruction* mem_obj);
  // Removes instructions queued up for removal during earlier processing
  // stages.
  spv_result_t RemoveDeadInstructions();

  // Returns the pointer-to-image and pointer-to-sampler types corresponding
  // the pointer-to-sampled-image-type. Creates them if needed, and updates
  // the def-use-manager.
  std::pair<Instruction*, Instruction*> GetPtrSamplerAndPtrImageTypes(
      Instruction& ptr_sample_image_type);

  // Cached from the IRContext. Valid while Process() is running.
  analysis::DefUseManager* def_use_mgr_ = nullptr;
  // Cached from the IRContext. Valid while Process() is running.
  analysis::TypeManager* type_mgr_ = nullptr;

  // Did processing modify the module?
  bool modified_ = false;
  Pass::Status Ok() {
    return modified_ ? Pass::Status::SuccessWithChange
                     : Pass::Status::SuccessWithoutChange;
  }

  // The first OpTypeSampledImage instruction in the module, if one exists.
  Instruction* first_sampled_image_type_ = nullptr;
  // An OpTypeSampler instruction, if one existed already, or if we created one.
  Instruction* sampler_type_ = nullptr;

  // The known types and module-scope values.
  // We use this to know when a new such value was created.
  std::unordered_set<uint32_t> known_globals_;

  // Combined types.  The known combined sampled-image type,
  // and recursively pointers or arrays of them.
  std::unordered_set<uint32_t> combined_types_;
  // The pre-existing types this pass should remove: pointer to
  // combined type, array of combined type, pointer to array of combined type.
  std::vector<uint32_t> combined_types_to_remove_;

  // Remaps a combined-kind type to corresponding sampler-kind and image-kind
  // of type.
  struct TypeRemapInfo {
    // The instruction for the combined type, pointer to combined type,
    // or point to array of combined type.
    Instruction* combined_kind_type;
    // The corresponding image type, with the same shape of indirection as the
    // combined_kind_type.
    Instruction* image_kind_type;
    // The corresponding sampler type, with the same shape of indirection as the
    // combined_kind_type.
    Instruction* sampler_kind_type;
  };
  // Maps the ID of a combined-image-sampler type kind to its corresponding
  // split parts.
  std::unordered_map<uint32_t, TypeRemapInfo> type_remap_;

  // Returns the image-like and sampler-like types of the same indirection shape
  // as the given combined-like type.  If combined_kind_type is not a type
  // or a pointer to one, then returns a pair of null pointer.
  // Either both components are non-null, or both components are null.
  std::pair<Instruction*, Instruction*> SplitType(
      Instruction& combined_kind_type);

  struct RemapValueInfo {
    // The original memory object for the combined entity.
    Instruction* combined_mem_obj = nullptr;
    // The instruction for the type of the original (combined) memory object.
    Instruction* combined_mem_obj_type = nullptr;
  };

  // Maps the ID of a memory object declaration for a combined texture+sampler
  // to remapping information about that object.
  std::unordered_map<uint32_t, RemapValueInfo> remap_info_;
  // The instructions added to remap_info_, in the order they were added.
  std::vector<Instruction*> ordered_objs_;

  // The instructions to be removed.
  std::vector<Instruction*> dead_;
  size_t num_to_delete_ = 0;  // TODO(dneto) remove this once arrays are handled
};
}  // namespace opt
}  // namespace spvtools
#endif  // LIBSPIRV_OPT_SPLIT_COMBINED_IMAGE_SAMPLER_PASS_H_
