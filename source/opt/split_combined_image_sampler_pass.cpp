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

  def_use_mgr_ = nullptr;
  type_mgr_ = nullptr;

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
      sampler_type_->InsertBefore(first_type_val);
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
    def_use_mgr_->AnalyzeInstDefUse(sampler_type_);
    type_mgr_->RegisterType(sampler_type_id, analysis::Sampler());
  }
  return SPV_SUCCESS;
}

spv_result_t SplitCombinedImageSamplerPass::RemapVars() {
  for (Instruction* var : ordered_objs_) {
    CHECK_STATUS(RemapVar(var));
  }
  return SPV_SUCCESS;
}

spv_result_t SplitCombinedImageSamplerPass::RemapVar(Instruction* var) {
  // Create an image variable, and a sampler variable.
  InstructionBuilder builder(context(), var, IRContext::kAnalysisDefUse);
  auto& info = remap_info_[var->result_id()];
  assert(info.var_id == var->result_id());
  assert(def_use_mgr_->GetDef(sampler_type_->result_id()));

  // Create the pointer types as needed.
  uint32_t sampler_ptr_ty_id = type_mgr_->FindPointerToType(
      sampler_type_->result_id(), spv::StorageClass::UniformConstant);
  uint32_t image_ptr_ty_id = type_mgr_->FindPointerToType(
      info.image_type, spv::StorageClass::UniformConstant);

  // By default they were created at the end of the types-and-global-vars list.
  // Move them to just after the pointee type declaration.
  auto* sampler_ptr_ty = def_use_mgr_->GetDef(sampler_ptr_ty_id);
  auto* image_ptr_ty = def_use_mgr_->GetDef(image_ptr_ty_id);
  auto* image_ty = def_use_mgr_->GetDef(info.image_type);

  sampler_ptr_ty->InsertAfter(sampler_type_);
  image_ptr_ty->InsertAfter(image_ty);

  // Create the variables.
  Instruction* sampler_var =
      builder.AddVariable(sampler_ptr_ty_id, SpvStorageClassUniformConstant);
  Instruction* image_var =
      builder.AddVariable(image_ptr_ty_id, SpvStorageClassUniformConstant);

  // SPIR-V has a Data rule:
  //  > All OpSampledImage instructions, or instructions that load an image or
  //  > sampler reference, must be in the same block in which their Result <id>
  //  > are consumed.
  //
  // Assuming that rule is honoured, the load is in the same block as the
  // operation using the sampled image that was loaded. So it's ok to load
  // the separate image and texture sampler, and also to create the combined
  // sampled image from them, all in the same basic block.

  struct Use {
    Instruction* user;
    uint32_t index;
  };
  std::vector<Use> uses;
  def_use_mgr_->ForEachUse(var, [&](Instruction* user, uint32_t use_index) {
    uses.push_back({user, use_index});
  });

  for (auto& use : uses) {
    switch (use.user->opcode()) {
      case spv::Op::OpLoad: {
        if (use.index != 2)
          return Fail() << "variable used as non-pointer index " << use.index
                        << " on load" << *use.user;
        Instruction* load = use.user;
        builder.SetInsertPoint(load);
        auto* image = builder.AddLoad(info.image_type, image_var->result_id());
        auto* sampler = builder.AddLoad(sampler_type_->result_id(),
                                        sampler_var->result_id());
        auto* sampled_image = builder.AddSampledImage(
            info.sampled_image_type, image->result_id(), sampler->result_id());
        this->def_use_mgr_->ForEachUse(
            load, [&](Instruction* user, uint32_t index) {
              user->SetOperand(index, {sampled_image->result_id()});
            });
        dead_.push_back(load);
        break;
      }
      case spv::Op::OpDecorate: {
        if (use.index != 0)
          return Fail() << "variable used as non-target index " << use.index
                        << " on decoration: " << *use.user;
        builder.SetInsertPoint(use.user);
        spv::Decoration deco{use.user->GetSingleWordInOperand(1)};
        std::vector<uint32_t> literals;
        for (uint32_t i = 2; i < use.user->NumInOperands(); i++) {
          literals.push_back(use.user->GetSingleWordInOperand(i));
        }
        builder.AddDecoration(image_var->result_id(), deco, literals);
        builder.AddDecoration(sampler_var->result_id(), deco, literals);
        dead_.push_back(use.user);
        break;
      }
      default:
        std::cout << "unhandled user: " << *use.user << std::endl;
    }
  }

  dead_.push_back(var);
  return SPV_SUCCESS;
}

spv_result_t SplitCombinedImageSamplerPass::RemoveDeadInstructions() {
  for (Instruction* inst : dead_) {
    def_use_mgr_->ClearInst(inst);
    // TODO delete dead type from type manager.
  }
  for (Instruction* inst : dead_) {
    inst->RemoveFromList();
    delete inst;
  }
  ordered_objs_.clear();
  dead_.clear();
  return SPV_SUCCESS;
}

}  // namespace opt
}  // namespace spvtools
