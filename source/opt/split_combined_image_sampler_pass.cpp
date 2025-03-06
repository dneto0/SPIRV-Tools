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

#include <algorithm>
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
  if (combined_types_to_remove_.empty()) {
    return Ok();
  }

  CHECK(RemapVars());
  CHECK(RemapFunctions());
  CHECK(RemoveDeadInstructions());

  def_use_mgr_ = nullptr;
  type_mgr_ = nullptr;

  return Ok();
}

spvtools::DiagnosticStream SplitCombinedImageSamplerPass::Fail() {
  return std::move(
      spvtools::DiagnosticStream({}, consumer(), "", SPV_ERROR_INVALID_BINARY)
      << "split-combined-image-sampler: ");
}

void SplitCombinedImageSamplerPass::FindCombinedTextureSamplers() {
  for (auto& inst : context()->types_values()) {
    RegisterGlobal(inst.result_id());
    switch (inst.opcode()) {
      case spv::Op::OpTypeSampler:
        // Note: The "if" should be redundant because valid modules can't have
        // duplicate sampler types.
        if (!sampler_type_) {
          sampler_type_ = &inst;
        }
        break;

      case spv::Op::OpTypeSampledImage:
        if (!first_sampled_image_type_) {
          first_sampled_image_type_ = &inst;
        }
        combined_types_.insert(inst.result_id());
        break;

      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeRuntimeArray: {
        auto pointee_id = inst.GetSingleWordInOperand(0);
        if (combined_types_.find(pointee_id) != combined_types_.end()) {
          combined_types_.insert(inst.result_id());
          combined_types_to_remove_.push_back(inst.result_id());
        }
      } break;

      case spv::Op::OpTypePointer: {
        auto sc =
            static_cast<spv::StorageClass>(inst.GetSingleWordInOperand(0));
        if (sc == spv::StorageClass::UniformConstant) {
          auto pointee_id = inst.GetSingleWordInOperand(1);
          if (combined_types_.find(pointee_id) != combined_types_.end()) {
            combined_types_.insert(inst.result_id());
            combined_types_to_remove_.push_back(inst.result_id());
          }
        }
      } break;

      case spv::Op::OpVariable:
        if (combined_types_.find(inst.type_id()) != combined_types_.end()) {
          ordered_objs_.push_back(&inst);
          auto& info = remap_info_[inst.result_id()];
          info.combined_mem_obj = &inst;
          info.combined_mem_obj_type = def_use_mgr_->GetDef(inst.type_id());
        }
        break;

      default:
        break;
    }
  }
}

Instruction* SplitCombinedImageSamplerPass::GetSamplerType() {
  if (!sampler_type_) {
    analysis::Sampler s;
    uint32_t sampler_type_id = type_mgr_->GetTypeInstruction(&s);
    sampler_type_ = def_use_mgr_->GetDef(sampler_type_id);
    assert(first_sampled_image_type_);
    sampler_type_->InsertBefore(first_sampled_image_type_);
    RegisterGlobal(sampler_type_->result_id());
    modified_ = true;
  }
  return sampler_type_;
}

spv_result_t SplitCombinedImageSamplerPass::RemapVars() {
  for (Instruction* mem_obj : ordered_objs_) {
    CHECK_STATUS(RemapVar(mem_obj));
  }
  return SPV_SUCCESS;
}

std::pair<Instruction*, Instruction*> SplitCombinedImageSamplerPass::SplitType(
    Instruction& combined_kind_type) {
  if (auto where = type_remap_.find(combined_kind_type.result_id());
      where != type_remap_.end()) {
    auto& type_remap = where->second;
    return {type_remap.image_kind_type, type_remap.sampler_kind_type};
  }

  switch (combined_kind_type.opcode()) {
    case spv::Op::OpTypeSampledImage: {
      auto* image_type =
          def_use_mgr_->GetDef(combined_kind_type.GetSingleWordInOperand(0));
      auto* sampler_type = GetSamplerType();
      type_remap_[combined_kind_type.result_id()] = {&combined_kind_type,
                                                     image_type, sampler_type};
      return {image_type, sampler_type};
      break;
    }
    case spv::Op::OpTypePointer: {
      auto sc = static_cast<spv::StorageClass>(
          combined_kind_type.GetSingleWordInOperand(0));
      if (sc == spv::StorageClass::UniformConstant) {
        auto* pointee =
            def_use_mgr_->GetDef(combined_kind_type.GetSingleWordInOperand(1));
        auto [image_pointee, sampler_pointee] = SplitType(*pointee);
        if (image_pointee && sampler_pointee) {
          auto make_pointer = [&](Instruction* pointee) {
            uint32_t ptr_id = type_mgr_->FindPointerToType(
                pointee->result_id(), spv::StorageClass::UniformConstant);
            auto* ptr = def_use_mgr_->GetDef(ptr_id);
            if (!IsKnownGlobal(ptr_id)) {
              // The pointer type was created at the end. Put it right after the
              // pointee.
              ptr->InsertBefore(pointee);
              pointee->InsertBefore(ptr);
              RegisterGlobal(ptr_id);
              def_use_mgr_->AnalyzeInstUse(pointee);
              modified_ = true;
            }
            return ptr;
          };
          auto* ptr_image = make_pointer(image_pointee);
          auto* ptr_sampler = make_pointer(sampler_pointee);
          type_remap_[combined_kind_type.result_id()] = {
              &combined_kind_type, ptr_image, ptr_sampler};
          num_to_delete_++;
          // dead_.push_back(&combined_kind_type); // Schedule for deletion.
          return {ptr_image, ptr_sampler};
        }
      }
      break;
    }
    // TODO(dneto): handle arrays
    default:
      break;
  }
  return {nullptr, nullptr};
}

spv_result_t SplitCombinedImageSamplerPass::RemapVar(Instruction* mem_obj) {
  InstructionBuilder builder(context(), mem_obj, IRContext::kAnalysisDefUse);
  // Create an image variable, and a sampler variable.
  auto& info = remap_info_[mem_obj->result_id()];

  // Create the variables.
  auto [ptr_image_ty, ptr_sampler_ty] = SplitType(*info.combined_mem_obj_type);
  if (!ptr_image_ty || !ptr_sampler_ty) {
    return Fail() << "unhandled case: array-of-combined-image-sampler";
  }
  Instruction* sampler_var = builder.AddVariable(
      ptr_sampler_ty->result_id(), SpvStorageClassUniformConstant);
  Instruction* image_var = builder.AddVariable(ptr_image_ty->result_id(),
                                               SpvStorageClassUniformConstant);
  modified_ = true;
  return RemapUses(mem_obj, image_var, sampler_var);
}

spv_result_t SplitCombinedImageSamplerPass::RemapUses(
    Instruction* combined, Instruction* image_part, Instruction* sampler_part) {
  // The insertion point should be updated before using this builder.
  // We needed *something* here.
  InstructionBuilder builder(context(), combined, IRContext::kAnalysisDefUse);

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
  def_use_mgr_->ForEachUse(combined,
                           [&](Instruction* user, uint32_t use_index) {
                             uses.push_back({user, use_index});
                           });

  for (auto& use : uses) {
    switch (use.user->opcode()) {
      case spv::Op::OpLoad: {
        if (use.index != 2)
          return Fail() << "variable used as non-pointer index " << use.index
                        << " on load" << *use.user;
        Instruction* load = use.user;

        auto pointee_ty_id = [&](Instruction* ptr_value) {
          auto* ptr_ty = def_use_mgr_->GetDef(ptr_value->type_id());
          assert(ptr_ty->opcode() == spv::Op::OpTypePointer);
          return ptr_ty->GetSingleWordInOperand(1);
        };

        builder.SetInsertPoint(load);
        auto* image =
            builder.AddLoad(pointee_ty_id(image_part), image_part->result_id());
        auto* sampler = builder.AddLoad(pointee_ty_id(sampler_part),
                                        sampler_part->result_id());
        auto* sampled_image = builder.AddSampledImage(
            load->type_id(), image->result_id(), sampler->result_id());
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
        builder.AddDecoration(image_part->result_id(), deco, literals);
        builder.AddDecoration(sampler_part->result_id(), deco, literals);
        // TODO(dneto): RelaxedPrecision?
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
        use.user->SetOperand(use.index, {image_part->result_id()});
        use.user->InsertOperand(
            use.user->NumOperands(),
            {SPV_OPERAND_TYPE_ID, {sampler_part->result_id()}});
        break;
      }
      case spv::Op::OpName:
        // TODO(dneto): synthesize names for the remapped vars.
        dead_.push_back(use.user);
        break;
      default:
        // TODO(dneto): OpFunctionCall
        // TODO(dneto): OpAccessChain, for arrays
        return Fail() << "unhandled user: " << *use.user;
    }
  }
  // We've added new uses of the new variables.
  def_use_mgr_->AnalyzeInstUse(image_part);
  def_use_mgr_->AnalyzeInstUse(sampler_part);

  dead_.push_back(combined);
  return SPV_SUCCESS;
}

spv_result_t SplitCombinedImageSamplerPass::RemapFunctions() {
  // Remap function types. A combined type can appear as a parameter, but not as
  // the return type.
  {
    std::unordered_set<Instruction*> reanalyze_set;
    for (auto& inst : context()->types_values()) {
      if (inst.opcode() == spv::Op::OpTypeFunction) {
        analysis::Function* f_ty =
            type_mgr_->GetType(inst.result_id())->AsFunction();
        std::vector<const analysis::Type*> new_params;
        for (const auto* param_ty : f_ty->param_types()) {
          const auto param_ty_id = type_mgr_->GetId(param_ty);
          if (combined_types_.find(param_ty_id) != combined_types_.end()) {
            auto* param_type = def_use_mgr_->GetDef(param_ty_id);
            auto [image_type, sampler_type] = SplitType(*param_type);
            assert(image_type);
            assert(sampler_type);
            // The image and sampler types must already exist, so there is no
            // need to move them to the right spot.
            new_params.push_back(type_mgr_->GetType(image_type->result_id()));
            new_params.push_back(type_mgr_->GetType(sampler_type->result_id()));
          } else {
            new_params.push_back(param_ty);
          }
        }
        if (new_params.size() != f_ty->param_types().size()) {
          // Replace this type.
          analysis::Function new_f_ty(f_ty->return_type(), new_params);
          const uint32_t new_f_ty_id = type_mgr_->GetTypeInstruction(&new_f_ty);
          def_use_mgr_->ForEachUse(&inst,
                                   [&](Instruction* user, uint32_t use_index) {
                                     user->SetOperand(use_index, {new_f_ty_id});
                                     reanalyze_set.insert(user);
                                   });
          dead_.push_back(&inst);

          reanalyze_set.insert(def_use_mgr_->GetDef(new_f_ty_id));
          // Reanalyze the non-combined parameter types, and the return type.
          auto* new_f_ty_inst = def_use_mgr_->GetDef(new_f_ty_id);
          new_f_ty_inst->ForEachId([&](const uint32_t* param_id_ptr) {
            reanalyze_set.insert(def_use_mgr_->GetDef(*param_id_ptr));
          });
        }
      }
    }
    for (auto* inst : reanalyze_set) {
      def_use_mgr_->AnalyzeInstDefUse(inst);
    }
  }

  // Rewite OpFunctionParameter in function definitions.
  for (Function& fn : *context()->module()) {
    std::vector<Instruction*> to_replace;
    fn.ForEachParam([&](Instruction* param) {
      auto param_ty_id = param->type_id();
      if (combined_types_.find(param_ty_id) != combined_types_.end()) {
        to_replace.push_back(param);
      }
    });
    if (to_replace.empty()) {
      continue;
    }
    auto next_to_replace = to_replace.begin();
    struct Replacement {
      Instruction* combined;
      Instruction* image;
      Instruction* sampler;
    };
    std::vector<Replacement> replacements;
    Function::RewriteParamFn rewriter =
        [&](std::unique_ptr<Instruction>&& from_param,
            std::back_insert_iterator<Function::ParamList>& appender) {
          auto param = std::move(from_param);
          if (param.get() == *next_to_replace) {
            auto* param_inst = param.release();
            auto* param_type = def_use_mgr_->GetDef(param->type_id());
            auto [image_type, sampler_type] = SplitType(*param_type);
            auto image_param = MakeUnique<Instruction>(
                context(), spv::Op::OpFunctionParameter,
                image_type->result_id(), context()->TakeNextId(),
                Instruction::OperandList{});
            auto sampler_param = MakeUnique<Instruction>(
                context(), spv::Op::OpFunctionParameter,
                sampler_type->result_id(), context()->TakeNextId(),
                Instruction::OperandList{});
            replacements.push_back(
                {param_inst, image_param.get(), sampler_param.get()});
            appender = std::move(image_param);
            appender = std::move(sampler_param);
            ++next_to_replace;
          } else {
            appender = std::move(param);
          }
        };
    fn.RewriteParams(rewriter);

    for (auto& replacement : replacements) {
      CHECK_STATUS(RemapUses(replacement.combined, replacement.image,
                             replacement.sampler));
    }
  }
  return SPV_SUCCESS;
}

spv_result_t SplitCombinedImageSamplerPass::RemoveDeadInstructions() {
  auto result = SPV_SUCCESS;
  for (auto dead_type_id : combined_types_to_remove_) {
    auto* ty = def_use_mgr_->GetDef(dead_type_id);
    dead_.push_back(ty);
    def_use_mgr_->ForEachUse(ty, [&](Instruction* user, uint32_t use_index) {
      if (user->opcode() == spv::Op::OpName) {
        dead_.push_back(user);
      }
    });
    CHECK_STATUS(result);
  }
  modified_ = modified_ || !dead_.empty();
  for (Instruction* inst : dead_) {
    def_use_mgr_->ClearInst(inst);
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
