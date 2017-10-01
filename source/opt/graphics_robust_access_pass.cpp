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

using spvtools::MakeUnique;

namespace spvtools {
namespace opt {

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
    for (auto& inst : block) {
      switch (inst.opcode()) {
        case SpvOpAccessChain:
        case SpvOpInBoundsAccessChain:
          ClampIndicesForAccessChain(&inst);
          break;
        default:
          break;
      }
    }
  }
  return _.modified;
}

void GraphicsRobustAccessPass::ClampIndicesForAccessChain(
    ir::Instruction* inst) {
  const uint32_t num_operands = inst->NumOperands();
  const uint32_t first_index = 3;
  for (uint32_t idx = first_index; idx < num_operands; ++idx) {
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
      auto import_inst = MakeUnique<ir::Instruction>(
          SpvOpExtInstImport, 0, _.glsl_insts_id,
          std::initializer_list<ir::Operand>{
              ir::Operand{SPV_OPERAND_TYPE_LITERAL_STRING, std::move(words)}});
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
      auto int_type_inst = MakeUnique<ir::Instruction>(
          SpvOpTypeInt, 0, result,
          std::initializer_list<ir::Operand>{
              ir::Operand{SPV_OPERAND_TYPE_LITERAL_INTEGER, {width}},
              ir::Operand{SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}});
      _.module->AddType(std::move(int_type_inst));
      assert(result);

      _.width_of_uint_type[result] = width;
    }
  }
  return result;
}

uint32_t GraphicsRobustAccessPass::GetUintValue(uint32_t type_id, uint64_t value) {
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
      auto constant_inst = MakeUnique<ir::Instruction>(
          SpvOpConstant, 0, result,
          std::initializer_list<ir::Operand>{
              ir::Operand{SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER, words},
          });
      _.module->AddGlobalValue(std::move(constant_inst));
  } else {
    result = where->second;
  }
  assert(result);
  return result;
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
        const ir::Operand& value_operand = inst.GetInOperand(0);
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
