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
  if (ordered_objs_.empty() && dead_.empty()) {
    return Pass::Status::SuccessWithoutChange;
  }

  CHECK(EnsureSamplerTypeAppearsFirst());
  CHECK(RemapFunctions());
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

      case spv::Op::OpTypePointer:
        if (static_cast<spv::StorageClass>(inst.GetSingleWordInOperand(0)) ==
            spv::StorageClass::UniformConstant) {
          auto* pointee = def_use_mgr_->GetDef(inst.GetSingleWordInOperand(1));
          if (pointee->opcode() == spv::Op::OpTypeSampledImage) {
            dead_.push_back(&inst);
          }
          // TODO(dneto): Delete pointer to array-of-sampled-image-type, and
          // pointer to runtime-array-of-sampled-image-type.
        }
        break;

      case spv::Op::OpVariable: {
        Instruction* ptr_ty = def_use_mgr_->GetDef(inst.type_id());
        if (Instruction* combined_ty =
                ptr_ty->GetVulkanResourcePointee(spv::Op::OpTypeSampledImage)) {
          ordered_objs_.push_back(&inst);
          auto& info = remap_info_[inst.result_id()];
          info.var_type = ptr_ty;
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
  if (!sampler_type_) {
    analysis::Sampler s;
    uint32_t sampler_type_id = type_mgr_->GetTypeInstruction(&s);
    if (sampler_type_id == 0) {
      return Fail() << "could not create a sampler type: ran out of IDs?";
    }
    sampler_type_ = def_use_mgr_->GetDef(sampler_type_id);
  }
  Instruction* first_type_val = &*(context()->types_values_begin());
  if (sampler_type_ != first_type_val) {
    sampler_type_->InsertBefore(first_type_val);
  }
  return SPV_SUCCESS;
}

spv_result_t SplitCombinedImageSamplerPass::RemapFunctions() {
  for (auto& inst : context()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeFunction) {
    }
  }
  return SPV_SUCCESS;
}

spv_result_t SplitCombinedImageSamplerPass::RemapVars() {
  for (Instruction* var : ordered_objs_) {
    CHECK_STATUS(RemapVar(var));
  }
  return SPV_SUCCESS;
}

std::pair<Instruction*, Instruction*>
SplitCombinedImageSamplerPass::GetPtrSamplerAndPtrImageTypes(
    Instruction& ptr_sampled_image_type) {
  assert(sampler_type_);
  assert(ptr_sampled_image_type.GetVulkanResourcePointee(
             spv::Op::OpTypeSampledImage) != nullptr);

  analysis::Type* ty = type_mgr_->GetType(ptr_sampled_image_type.result_id());
  analysis::Pointer* ptr_combined_ty = ty->AsPointer();
  assert(ptr_combined_ty);
  assert(ptr_combined_ty->storage_class() ==
         spv::StorageClass::UniformConstant);

  auto* pointee_ty = ptr_combined_ty->pointee_type();
  const analysis::SampledImage* combined_ty = pointee_ty->AsSampledImage();

  // TODO(dneto): handle array-of-combined, runtime-array-of-combined

  if (combined_ty) {
    // The image type must have already existed.
    auto* image_ty = def_use_mgr_->GetDef(
        type_mgr_->GetTypeInstruction(combined_ty->image_type()));

    // Create the pointer types as needed.
    uint32_t ptr_sampler_ty_id = type_mgr_->FindPointerToType(
        sampler_type_->result_id(), spv::StorageClass::UniformConstant);
    uint32_t ptr_image_ty_id = type_mgr_->FindPointerToType(
        image_ty->result_id(), spv::StorageClass::UniformConstant);

    // By default they were created at the end of the types-and-global-vars
    // list. Move them to just after the pointee type declaration.
    auto* ptr_sampler_ty = def_use_mgr_->GetDef(ptr_sampler_ty_id);
    auto* ptr_image_ty = def_use_mgr_->GetDef(ptr_image_ty_id);
    ptr_sampler_ty->InsertAfter(sampler_type_);
    ptr_image_ty->InsertAfter(image_ty);

    // Update cross-references.
    def_use_mgr_->AnalyzeInstUse(sampler_type_);
    def_use_mgr_->AnalyzeInstUse(ptr_sampler_ty);
    def_use_mgr_->AnalyzeInstUse(image_ty);
    def_use_mgr_->AnalyzeInstUse(ptr_image_ty);

    return {ptr_image_ty, ptr_sampler_ty};
  }
  return {};
}

spv_result_t SplitCombinedImageSamplerPass::RemapVar(Instruction* var) {
  // Create an image variable, and a sampler variable.
  InstructionBuilder builder(context(), var, IRContext::kAnalysisDefUse);
  auto& info = remap_info_[var->result_id()];
  assert(info.var_id == var->result_id());

  // Create the variables.
  auto [ptr_image_ty, ptr_sampler_ty] =
      GetPtrSamplerAndPtrImageTypes(*info.var_type);
  if (!ptr_image_ty) {
    return Fail() << "unhandled case: array-of-combined-image-sampler";
  }
  Instruction* sampler_var = builder.AddVariable(
      ptr_sampler_ty->result_id(), SpvStorageClassUniformConstant);
  Instruction* image_var = builder.AddVariable(ptr_image_ty->result_id(),
                                               SpvStorageClassUniformConstant);

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
        def_use_mgr_->AnalyzeInstUse(image);
        def_use_mgr_->AnalyzeInstUse(sampler);
        def_use_mgr_->AnalyzeInstUse(sampled_image);
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
      case spv::Op::OpEntryPoint: {
        // The entry point lists variables in the shader interface, i.e.
        // module-scope variables referenced by the static call tree rooted
        // at the entry point. (It can be a proper superset).  Before SPIR-V
        // 1.4, only Input and Output variables are listed; in 1.4 and later,
        // module-scope variables in all storage classes are listed.
        // If a combined image+sampler is listed by the entry point, then
        // the separated image and sampler variables should be.
        if (use.index < 3)
          return Fail() << "variable used in index " << use.index
                        << " instead of as an interface variable:" << *use.user;
        // Avoid moving the other IDs around, so we don't have to update their
        // uses in the def_use_mgr_.
        use.user->SetOperand(use.index, {image_var->result_id()});
        use.user->InsertOperand(
            use.user->NumOperands(),
            {SPV_OPERAND_TYPE_ID, {sampler_var->result_id()}});
        break;
      }
      case spv::Op::OpName:
        // TODO(dneto): synthesize names for the remapped vars.
        dead_.push_back(use.user);
        break;
      default:
        // TODO(dneto): OpFunctionCall
        std::cout << "unhandled user: " << *use.user << std::endl;
    }
  }
  // We've added new uses of the new variables.
  def_use_mgr_->AnalyzeInstUse(image_var);
  def_use_mgr_->AnalyzeInstUse(sampler_var);

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
