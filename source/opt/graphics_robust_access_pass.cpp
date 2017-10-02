// Copyright (c) 2017 Google Inc.
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

// This injects code in a graphics shader to implement guarantees satisfying
// Vulkan's robustBufferAcces rules.  Robust access rules permit an
// out-of-bounds accesses to be redirected to an access of the same type
// (load, store, etc.) but within the same root object.
//
// We assume baseline functionality in Vulkan, i.e. the module uses
// logical addressing mode, without VK_KHR_variable_pointers.
//
//    - Logical addressing mode implies:
//      - Each root pointer (a pointer that exists other than by the
//        execution of a shader instruction) is the result of an OpVariable.
//
//      - Instructions that result in pointers are:
//          OpVariable
//          OpAccessChain
//          OpInBoundsAccessChain
//          OpFunctionParameter
//          OpImageTexelPointer
//          OpCopyObject
//
//      - Instructions that use a pointer are:
//          OpLoad
//          OpStore
//          OpAccessChain
//          OpInBoundsAccessChain
//          OpFunctionCall
//          OpImageTexelPointer
//          OpCopyMemory
//          OpCopyObject
//          all OpAtomic* instructions
//
// We classify pointer-users into:
//  - Accesses:
//    - OpLoad
//    - OpStore
//    - OpAtomic*
//    - OpCopyMemory
//
//  - Address calculations:
//    - OpAccessChain
//    - OpInBoundsAccessChain
//
//  - Pass-through:
//    - OpFunctionCall
//    - OpFunctionParameter
//    - OpCopyObject
//
// The strategy is:
//
//WIP:
//TODO:
//
//  - Validate that pointers are only used by the instructions as above.
//    (Or rely on an external validator?)
//
//  - Clamp indices contributing to address calculations.
//    The valid range depends on the targeted type at each index,
//    and sometimes based queries on the object itself.
//
//  - Assume exhaustive inlining has occured, so function calls are not
//    accesses.
//
//  - Handle pass through of pointers via OpCopyObject

#include "graphics_robust_access_pass.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <utility>

#include "spirv/1.2/spirv.h"

#include "diagnostic.h"

#include "function.h"
#include "make_unique.h"
#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

using ir::BasicBlock;
using ir::Instruction;
using ir::Operand;
using spvtools::MakeUnique;

GraphicsRobustAccessPass::GraphicsRobustAccessPass() : _(nullptr) {}

libspirv::DiagnosticStream GraphicsRobustAccessPass::Fail() {
  _.failed = true;
  // We don't really have a position, and we'll ignore the result.
  return std::move(
      libspirv::DiagnosticStream({}, consumer(), SPV_ERROR_INVALID_BINARY)
      << name() << ": ");
}

spv_result_t GraphicsRobustAccessPass::ProcessCurrentModule() {
  if (_.module->HasCapability(SpvCapabilityVariablePointers))
    return Fail() << "Can't process module with VariablePointers capability";

  {
    const auto addressing_model =
        _.module->GetMemoryModel()->GetSingleWordOperand(0);
    if (addressing_model != SpvAddressingModelLogical)
      return Fail() << "Can't process module with addressing model other than "
                       "Logical.  Found "
                    << int(addressing_model);
  }

  ProcessFunction fn = [this](ir::Function* f) { return ProcessAFunction(f); };
  _.modified |= ProcessReachableCallTree(fn, _.module);

  // Need something here.  It's the price we pay for easier failure paths.
  return SPV_SUCCESS;
}

bool GraphicsRobustAccessPass::ProcessAFunction(ir::Function* function) {
  // Ensure that all pointers computed inside a function are within bounds.
  for (auto& block : *function) {
    for (auto inst_iter = block.begin(); inst_iter != block.end();
         ++inst_iter) {
      switch (inst_iter->opcode()) {
        case SpvOpAccessChain:
        case SpvOpInBoundsAccessChain:
          ClampIndicesForAccessChain(&inst_iter);
          break;
        default:
          break;
      }
    }
  }
  return _.modified;
}

void GraphicsRobustAccessPass::ClampIndicesForAccessChain(
    BasicBlock::iterator* inst_iter_ptr) {
  Instruction& inst = **inst_iter_ptr;
  const uint32_t num_operands = inst.NumOperands();

  uint32_t ptr_id = inst.GetSingleWordOperand(3);
  const Instruction* ptr_inst = GetDef(ptr_id);
  const Instruction* ptr_type = GetDef(ptr_inst->type_id());
  Instruction* pointee_type = GetDef(ptr_type->GetSingleWordOperand(3));
  const uint32_t first_index = 4;

  auto clamp_index = [&inst_iter_ptr, &inst, this](uint32_t operand_index,
                                                   uint32_t old_value_id,
                                                   uint32_t max_value_id) {
    auto umax_inst = MakeUmaxInst(old_value_id, max_value_id);

    // Subtract 2 from the index to account for result id and type id.
    inst.SetInOperand(operand_index - 2, {umax_inst->result_id()});

    // Insert the new instuction
    BasicBlock::iterator& inst_iter = *inst_iter_ptr;
    inst_iter = inst_iter.InsertBefore(std::move(umax_inst));
    // Get back to the AccessChain instruction.
    ++inst_iter;
  };

  // Walk the indices, replacing indices with a clamped value, and updating
  // the pointee_type.
  for (uint32_t idx = first_index; idx < num_operands; ++idx) {
    const uint32_t index_id = inst.GetSingleWordOperand(idx);
    const Instruction* index_inst = GetDef(index_id);
    uint32_t index_type_id = index_inst->type_id();
    switch (pointee_type->opcode()) {
      case SpvOpTypeMatrix:  // Use column count
      case SpvOpTypeVector:  // Use component count
      {
        const uint32_t max_index_value_id = GetUintValue(
            index_type_id, pointee_type->GetSingleWordOperand(2) - 1);
        pointee_type = GetDef(pointee_type->GetSingleWordOperand(1));
        clamp_index(idx, index_id, max_index_value_id);
      } break;

      case SpvOpTypeArray: {
        // The array length could be a spec constant.  For now only handle
        // the case where it's a constant.
        // TODO(dneto): Handle op-spec constant case.
        const Instruction* array_len =
            GetDef(pointee_type->GetSingleWordOperand(2));
        if (array_len->opcode() != SpvOpConstant) {
          Fail() << "Array type with id " << array_len->result_id()
                 << " uses a length which is not an OpConstant.  Found opcode "
                 << array_len->opcode()
                 << " instead.  The OpSpecConstant case is not handled yet.";
          return;
        }
        auto which_type_iter = _.width_of_uint_type.find(array_len->type_id());
        if (which_type_iter == _.width_of_uint_type.end()) {
          Fail() << "Array length value with id " << array_len->result_id()
                 << " is of type " << array_len->type_id()
                 << " which is not an integer type of less than 64 bits";
          return;
        }
        uint64_t len = GetUintValueFromConstant(*array_len);
        const uint32_t max_index_value_id =
            GetUintValue(index_type_id, len - 1);
        pointee_type = GetDef(pointee_type->GetSingleWordOperand(1));
        clamp_index(idx, index_id, max_index_value_id);
      } break;

      case SpvOpTypeStruct: {
        if (index_inst->opcode() != SpvOpConstant) {
          Fail() << "Struct index with id " << index_inst->result_id()
                 << " in access chain " << inst.result_id()
                 << " is not an OpConstant.  Found opcode "
                 << index_inst->opcode() << " instead.";
          return;
        }
        auto which_type_iter = _.width_of_uint_type.find(index_inst->type_id());
        if (which_type_iter == _.width_of_uint_type.end()) {
          Fail()
              << "Struct index with id " << index_inst->result_id()
              << " in access chain " << inst.result_id() << " is of type "
              << index_inst->type_id()
              << " which is not an unsigned integer type of less than 64 bits";
          return;
        }

        const uint32_t num_members = pointee_type->NumInOperands();
        const auto index_value = GetUintValueFromConstant(*index_inst);
        if (index_value >= num_members) {
          Fail() << "In access chain " << inst.result_id()
                 << ", member index value " << index_value
                 << " is too large for struct type with id "
                 << pointee_type->result_id();
          return;
        }
        pointee_type = GetDef(pointee_type->GetSingleWordInOperand(
            static_cast<uint32_t>(index_value)));
        // No need to clamp this index.  We just checked that it's valid.
      } break;

      case SpvOpTypeRuntimeArray:
        Fail() << " Unhandled runtime array ";
        pointee_type = GetDef(pointee_type->GetSingleWordOperand(1));
        return;

      default:
        Fail() << " Unhandled pointee type with opcode "
               << pointee_type->opcode();
    }
  }
}

uint32_t GraphicsRobustAccessPass::GetGlslInsts() {
  if (_.glsl_insts_id == 0) {
    // This string serves double-duty as raw data for a string and for a vector
    // of 32-bit words
    const char glsl[] = "GLSL.std.450\0\0\0\0";
    const size_t glsl_str_byte_len = 16;
    // Use an existing import if we can.
    for (auto& inst : _.module->ext_inst_imports()) {
      const auto& name_words = inst.GetInOperand(0).words;
      if (0 == std::strncmp(reinterpret_cast<const char*>(name_words.data()),
                            glsl, glsl_str_byte_len)) {
        _.glsl_insts_id = inst.result_id();
      }
    }
    if (_.glsl_insts_id == 0) {
      // Make a new import.
      _.modified = true;
      _.glsl_insts_id = _.next_id++;
      std::vector<uint32_t> words(4);
      std::memcpy(words.data(), glsl, glsl_str_byte_len);
      auto import_inst = MakeUnique<Instruction>(
          SpvOpExtInstImport, 0, _.glsl_insts_id,
          std::initializer_list<Operand>{
              Operand{SPV_OPERAND_TYPE_LITERAL_STRING, std::move(words)}});
      _.module->AddExtInstImport(std::move(import_inst));
    }
  }
  return _.glsl_insts_id;
}

uint32_t GraphicsRobustAccessPass::GetUintType(uint32_t width) {
  uint32_t& result = _.uint_type[width];
  if (result == 0) {
    // Find a prexisting type definition if it exists.
    for (auto& inst : _.module->types_values()) {
      if (inst.opcode() == SpvOpTypeInt &&
          inst.GetSingleWordOperand(1) == width &&
          inst.GetSingleWordOperand(2) == 0) {
        result = inst.result_id();
      }
      assert(result);
    }
    if (result == 0) {
      // Make a new declaration.
      _.modified = true;
      result = _.next_id++;
      auto int_type_inst = MakeUnique<Instruction>(
          SpvOpTypeInt, 0, result,
          std::initializer_list<Operand>{
              Operand{SPV_OPERAND_TYPE_LITERAL_INTEGER, {width}},
              Operand{SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}});
      _.module->AddType(std::move(int_type_inst));
      assert(result);

      _.width_of_uint_type[result] = width;
    }
  }
  return result;
}

uint32_t GraphicsRobustAccessPass::GetUintValue(uint32_t type_id,
                                                uint64_t value) {
  auto type_value = std::make_pair(type_id, value);
  auto where = _.uint_value.find(type_value);
  uint32_t result = 0;
  if (where == _.uint_value.end()) {
    // Make a new constant.
    _.modified = true;
    result = _.next_id++;
    // Construct the raw words.  Assume the type is at most 64 bits
    // wide.
    std::vector<uint32_t> words;
    words.push_back(static_cast<uint32_t>(value & 0xffff));
    if (_.width_of_uint_type[type_id] > 32) {
      words.push_back(static_cast<uint32_t>(value >> 32));
    }
    auto constant_inst = MakeUnique<Instruction>(
        SpvOpConstant, 0, result,
        std::initializer_list<Operand>{
            Operand{SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER, words},
        });
    _.module->AddGlobalValue(std::move(constant_inst));
  } else {
    result = where->second;
  }
  assert(result);
  return result;
}

uint64_t GraphicsRobustAccessPass::GetUintValueFromConstant(
    const Instruction& inst) {
  assert(inst.opcode() == OpConstant);
  assert(_.width_of_uint_type.find(inst.type_id()) !=
         _.width_of_uint_type.end());

  const auto& value_words = inst.GetInOperand(0).words;
  assert(1 <= value_words.size());
  assert(value_words.size() <= 2);
  uint64_t result = value_words[0];
  if (value_words.size() == 2) {
    result |= (uint64_t(value_words[1]) << 32);
  }
  return result;
}

std::unique_ptr<Instruction> GraphicsRobustAccessPass::MakeUmaxInst(
    uint32_t id0, uint32_t id1) {
  _.modified = true;
  auto umax_inst =
      MakeUnique<Instruction>(SpvOpExtInst, GetDef(id0)->type_id(), _.next_id++,
                              std::initializer_list<Operand>{
                                  Operand{SPV_OPERAND_TYPE_ID, {id0}},
                                  Operand{SPV_OPERAND_TYPE_ID, {id1}},
                              });
  return umax_inst;
}

void GraphicsRobustAccessPass::LoadUintTypeWidths() {
  for (auto& inst : _.module->types_values()) {
    if (inst.opcode() == SpvOpTypeInt && inst.GetSingleWordOperand(2) == 0) {
      const auto width = inst.GetSingleWordOperand(1);
      if (width <= 64) {
        _.width_of_uint_type[inst.result_id()] = width;
      }
    }
  }
}

void GraphicsRobustAccessPass::LoadUintValues() {
  for (auto& inst : _.module->types_values()) {
    if (inst.opcode() == SpvOpConstant) {
      const uint32_t type_id = inst.type_id();
      auto uint_type_iter = _.width_of_uint_type.find(type_id);
      if (uint_type_iter != _.width_of_uint_type.end()) {
        // This is an unsigned integer constant of up to 64 bits.
        // Copy its bits into |value|.
        uint64_t value = 0;
        const Operand& value_operand = inst.GetInOperand(0);
        assert(value_operand.words.size() <= 2);
        std::memcpy(&value, value_operand.words.data(),
                    value_operand.words.size() * sizeof(uint32_t));
        _.uint_value[std::make_pair(type_id, value)] = inst.result_id();
      }
    }
  }
}

Pass::Status GraphicsRobustAccessPass::Process(ir::Module* module) {
  _ = PerModuleState(module);
  LoadUintTypeWidths();
  LoadUintValues();

  ProcessCurrentModule();

  auto result = _.failed ? Status::Failure
                         : (_.modified ? Status::SuccessWithChange
                                       : Status::SuccessWithoutChange);

  _ = PerModuleState(nullptr);

  return result;
}

}  // namespace opt
}  // namespace spvtools
