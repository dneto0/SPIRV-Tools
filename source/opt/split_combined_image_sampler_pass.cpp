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

#include <cassert>
#include <iostream>
#include <memory>

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

#define CHECK_STATUS(cond)                         \
  {                                                \
    if (auto c = cond; c != SPV_SUCCESS) return c; \
  }

Pass::Status SplitCombinedImageSamplerPass::Process() {
  def_use_mgr_ = context()->get_def_use_mgr();
  type_mgr_ = context()->get_type_mgr();

  FindCombinedTextureSamplers();
  if (ordered_objs_.empty()) return Pass::Status::SuccessWithoutChange;

  CHECK(EnsureSamplerTypeAppearsFirst());
  CHECK(RemapVars());
  CHECK(RemoveDeadInstructions());

  return Pass::Status::SuccessWithChange;
}

spvtools::DiagnosticStream SplitCombinedImageSamplerPass::Fail() {
  return std::move(
      spvtools::DiagnosticStream({}, consumer(), "", SPV_ERROR_INVALID_BINARY)
      << "split-combined-image-sampler: ");
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
          ordered_objs_.push_back(&inst);
          auto& info = remap_info_[inst.result_id()];
          info.var_id = inst.result_id();
          info.sampled_image_type = combined_ty->result_id();
          info.image_type = def_use_mgr_->GetDef(info.sampled_image_type)
                                ->GetSingleWordInOperand(0);
        }
        break;
      }
    }
  }
}

spv_result_t SplitCombinedImageSamplerPass::EnsureSamplerTypeAppearsFirst() {
  // Put it at the start of the types-and-values list:
  // It depends on nothing, and other things will depend on it.
  Instruction* first_type_val = &*(context()->types_values_begin());
  if (sampler_type_) {
    if (sampler_type_ != first_type_val) {
      sampler_type_->RemoveFromList();
      std::unique_ptr<Instruction> inst(sampler_type_);
      first_type_val->InsertBefore(std::move(inst));
    }
  } else {
    InstructionBuilder builder(
        context(), first_type_val,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

    // Create it.
    uint32_t sampler_type_id = context()->TakeNextId();
    if (sampler_type_id == 0) {
      return Fail() << "ran out of IDs when creating a sampler type";
    }

    std::unique_ptr<Instruction> inst(new Instruction(
        context(), spv::Op::OpTypeSampler, 0, sampler_type_id, {}));

    sampler_type_ = builder.AddInstruction(std::move(inst));
    std::cout << " sampler type id " << sampler_type_->result_id() << std::endl;
    def_use_mgr_->AnalyzeInstDefUse(sampler_type_);
    type_mgr_->RegisterType(sampler_type_id, analysis::Sampler());
  }
  return SPV_SUCCESS;
}

spv_result_t SplitCombinedImageSamplerPass::RemapVars() {
  for (Instruction* var : ordered_objs_) {
    CHECK_STATUS(RemapVar(var));
  }
}

spv_result_t SplitCombinedImageSamplerPass::RemapVar(Instruction* var) {
  // Create an image variable, and a sampler variable.
  InstructionBuilder builder(context(), var, IRContext::kAnalysisDefUse);
  auto& info = remap_info_[var->result_id()];
  assert(info.var_id == var->result_id());
  std::cout << " samplare type id is " << sampler_type_->result_id()
            << std::endl;
  assert(def_use_mgr_->GetDef(sampler_type_->result_id()));
  Instruction* sampler_var = builder.AddVariable(
      sampler_type_->result_id(), SpvStorageClassUniformConstant);
  Instruction* image_var =
      builder.AddVariable(info.image_type, SpvStorageClassUniformConstant);

  // TODO: transfer decorations
  // TODO: update bindings

  def_use_mgr_->ForEachUser(var, [&](Instruction* user) {
    std::cout << "=uaer..." << std::endl;
    switch (user->opcode()) {
      case spv::Op::OpLoad: {
        Instruction* load = user;
        builder.SetInsertPoint(load);
        std::cout << "= load. image.." << std::endl;
        auto* image = builder.AddLoad(info.image_type, image_var->result_id());
        std::cout << "= load. sampler.." << std::endl;
        auto* sampler = builder.AddLoad(sampler_type_->result_id(),
                                        sampler_var->result_id());
        std::cout << "= combine.." << std::endl;
        auto* sampled_image = builder.AddSampledImage(
            info.sampled_image_type, image->result_id(), sampler->result_id());
        std::cout << "= RAUW.." << def_use_mgr_ << std::endl;
        def_use_mgr_->ForEachUse(load, [&](Instruction* user, uint32_t index) {
          std::cout << index << std::endl;
          user->SetOperand(index, {sampled_image->result_id()});
        });
        dead_.push_back(load);
        break;
      }
        // todo function call
    }
  });

  dead_.push_back(var);
}

spv_result_t SplitCombinedImageSamplerPass::RemoveDeadInstructions() {
  // TODO delete dead typefrom type manager.
  for (Instruction* inst : dead_) {
    def_use_mgr_->ClearInst(inst);
  }
  for (Instruction* inst : dead_) {
    inst->RemoveFromList();
    delete inst;
  }
  dead_.clear();
}

}  // namespace opt
}  // namespace spvtools
