// Copyright (c) 2018 Google LLC
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

#include "source/opt/split_combined_image_sampler_pass.h"

#include <iostream>

#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/util/make_unique.h"
#include "spirv/unified1/spirv.h"

namespace spvtools {
namespace opt {

Pass::Status SplitCombinedImageSamplerPass::Process() {
  def_use_mgr_ = context()->get_def_use_mgr();
  FindCombinedTextureSamplers();
  return ordered_objs_.empty() ? Pass::Status::SuccessWithoutChange
                               : Pass::Status::SuccessWithChange;
}

spvtools::DiagnosticStream SplitCombinedImageSamplerPass::Fail() {
  module_status_.failed = true;
  return std::move(
      spvtools::DiagnosticStream({}, consumer(), "", SPV_ERROR_INVALID_BINARY)
      << "split-combined-image-sampler: ");
}

void SplitCombinedImageSamplerPass::FindCombinedTextureSamplers() {
  for (auto& inst : context()->types_values()) {
    if (inst.opcode() != spv::Op::OpVariable) continue;
    Instruction* ptr_ty = def_use_mgr_->GetDef(inst.type_id());
    if (Instruction* combined_ty = ptr_ty->AsVulkanCombinedSampledImageType()) {
      ordered_objs_.push_back(inst.result_id());
      auto& info = remap_info_[inst.result_id()];
      info.mem_obj_decl = inst.result_id();
      info.sampled_image_type = combined_ty->result_id();
    }
  }
}

}  // namespace opt
}  // namespace spvtools
