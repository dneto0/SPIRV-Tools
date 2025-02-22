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

#define CHECK(cond)                                        \
  {                                                        \
    if (cond != SPV_SUCCESS) return Pass::Status::Failure; \
  }

Pass::Status SplitCombinedImageSamplerPass::Process() {
  def_use_mgr_ = context()->get_def_use_mgr();
  type_mgr_ = context()->get_type_mgr();

  FindCombinedTextureSamplers();
  if (ordered_objs_.empty()) return Pass::Status::SuccessWithoutChange;

  CHECK(EnsureSamplerTypeAppearsFirst());

  return Pass::Status::SuccessWithChange;
}

spvtools::DiagnosticStream SplitCombinedImageSamplerPass::Fail() {
  return std::move(
      spvtools::DiagnosticStream({}, consumer(), "", SPV_ERROR_INVALID_BINARY)
      << "split-combined-image-sampler: ");
}

spv_result_t SplitCombinedImageSamplerPass::EnsureSamplerTypeAppearsFirst() {
  if (sampler_type_) {
    sampler_type_->RemoveFromList();
  } else {
    // Create it.
    uint32_t sampler_type_id = context()->TakeNextId();
    if (sampler_type_id == 0) {
      return Fail() << "ran out of IDs when creating a sampler type";
    }

    sampler_type_ = new Instruction(context(), spv::Op::OpTypeSampler,
                                    sampler_type_id, 0, {});

    // Update analyses.
    def_use_mgr_->AnalyzeInstDefUse(sampler_type_);
    type_mgr_->RegisterType(sampler_type_id, analysis::Sampler());
  }

  // Put it at the start of the types-and-values list:
  // It depends on nothing, and other things will depend on it.
  std::unique_ptr<Instruction> inst(sampler_type_);
  context()->types_values_begin()->InsertBefore(std::move(inst));

  return SPV_SUCCESS;
}

void SplitCombinedImageSamplerPass::FindCombinedTextureSamplers() {
  for (auto& inst : context()->types_values()) {
    switch (inst.opcode()) {
      case spv::Op::OpTypeSampler:
        // Note: In any case, valid modules can't have duplicate sampler types.
        sampler_type_ = &inst;
        break;

      case spv::Op::OpVariable: {
        Instruction* ptr_ty = def_use_mgr_->GetDef(inst.type_id());
        if (Instruction* combined_ty =
                ptr_ty->GetVulkanResourcePointee(spv::Op::OpTypeSampledImage)) {
          ordered_objs_.push_back(inst.result_id());
          auto& info = remap_info_[inst.result_id()];
          info.mem_obj_decl = inst.result_id();
          info.sampled_image_type = combined_ty->result_id();
        }
        break;
      }
    }
  }
}

}  // namespace opt
}  // namespace spvtools
