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

#include "spirv/1.2/spirv.h"

#include "diagnostic.h"

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

  // Need something here.  It's the price we pay for easier failure paths.
  return SPV_SUCCESS;
}

Pass::Status GraphicsRobustAccessPass::Process(ir::Module* module) {
  _ = PerModuleState(module);

  ProcessCurrentModule();

  auto result = _.failed ? Status::Failure
                         : (_.modified ? Status::SuccessWithChange
                                       : Status::SuccessWithoutChange);

  _ = PerModuleState(nullptr);

  return result;
}  // namespace opt

}  // namespace opt
}  // namespace spvtools
