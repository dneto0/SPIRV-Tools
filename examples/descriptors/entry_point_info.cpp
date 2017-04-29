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

// TODO(dneto): Capture variables referenced in entry point function itself
// TODO(dneto): Cover mentions in: OpVariable
// TODO(dneto): Cover mentions in: OpAccessChain, OpInBoundsAccessChain
// TODO(dneto): Cover mentions in: OpFunctionParamter
// TODO(dneto): Cover mentions in: OpImageTexelPointer
// TODO(dneto): Cover mentions in: OpCopyObject
// TODO(dneto): Cover mentions in: OpAtomic*
// TODO(dneto): Variables referenced in call stack.

#include "entry_point_info.h"

#include <cassert>
#include <unordered_map>
#include <utility>
#include <vector>

#include "spirv/1.1/spirv.h"
#include "spirv-tools/libspirv.h"

using spirv_example::EntryPointInfo;
using spirv_example::Descriptor;
using spirv_example::Descriptors;

namespace {

// Returns a string for the literal string logical operand starting at the
// given word.  assumes it has a terminating null as required.
std::string StringFromWords(const uint32_t* words) {
  return reinterpret_cast<const char*>(words);
}

// Returns true if the given storage class can contain variables that have
// descriptors.
bool IsStorageClassWithDescriptors(SpvStorageClass sc) {
  switch (sc) {
    case SpvStorageClassUniformConstant:
    case SpvStorageClassUniform:
    case SpvStorageClassStorageBuffer:
      return true;
    default:
      break;
  }
  return false;
}

// A collector holds accumulated information about all entry points in the
// module.
class Collector {
 public:
  explicit Collector(std::vector<EntryPointInfo>* entry_points)
      : entry_points_(entry_points), current_function_ {
    entry_points_->clear();
  }

  spv_result_t HandleInstruction(const spv_parsed_instruction_t& inst) {
    switch (inst.opcode) {
      case SpvOpEntryPoint:
        entry_points_->push_back(
            EntryPointInfo{StringFromWords(inst.words + 3)});
        break;
      case SpvOpDecoration: {
        const auto target = inst.words[1];
        if (inst.num_words == 4) {
          const auto number = inst.words[3];
          switch (inst.words[2]) {
            case SpvDecorationDescriptorSet:
            case SpvDecorationBinding:
              SaveDescriptorInfo(target, inst.words[2], number);
              break;
            default:
              break;
          }
        }
      } break;
      default:
    }
    return SPV_SUCCESS;
  }

  // Saves the fact about the descriptor set or binding information for
  // the given target id.
  void SaveDescriptor(uint32_t target, SpvDecoration decoration,
                      uint32_t number) {
    auto d& = descriptors_[target];
    if (decoration == SpvDecorationDescriptorSet) {
      d.set = number;
    } else if (decoration == SpvDecorationBinding) {
      d.binding = number;
    }
  }

 private:
  // The accumulated entry point information.
  std::vector<EntryPointInfo>* entry_points_;

  // The Id of the current function.  We have seen its OpFunction instruction
  // but not its OpFunctionEnd instruction.  This is 0 when there is no current
  // function.
  uint32_t current_function_;

  // Maps the Id of a function to the directly referenced descriptors.
  std::unordered_map<uint32_t, Descriptors> uses_;

  // Maps an Id to the descriptor decorated on it.
  std::unordered_map<uint32_t, Descriptors> descriptors_;
};

// Binary parser handle-instruction callback.
// Captures necessary information from an instruction, assuming they are
// seen in order.
spv_result_t HandleInstruction(
    void* user_data, const spv_parsed_instruction_t* parsed_instruction) {
  assert(user_data);
  auto collector = static_cast<Collector*>(user_data);
  return collector->HandleInstruction(*parsed_instruction);
}
}

namespace spirv_example {

spv_result_t GetEntryPointInfo(const spv_const_context context,
                               const uint32_t* words, size_t num_words,
                               std::vector<EntryPointInfo>* entry_points,
                               spv_diagnostic* diagnostic) {
  if (!entry_points) return SPV_ERROR_INVALID_POINTER;
  Collector collector(entry_points);
  auto status = spvBinaryParse(context, &collector, words, num_words, nullptr,
                               HandleInstruction, diagnostic);
  return status;
}

}  // namespace spirv_example
