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

// A collector holds accumulated information about all entry points in the
// module.  It assumes the module is valid.
//
// It assumes that a descriptor will have both a DesciptorSet and Binding
// decoration.
class Collector {
 public:
  explicit Collector(std::vector<EntryPointInfo>* entry_points)
      : entry_points_(entry_points), current_function_(0) {
    entry_points_->clear();
  }

  spv_result_t HandleInstruction(const spv_parsed_instruction_t& inst) {
    switch (inst.opcode) {
      case SpvOpEntryPoint:
	entry_point_map_[inst.words[2]] = entry_points_->size();
        entry_points_->push_back(
            EntryPointInfo{StringFromWords(inst.words + 3)});
        break;
      case SpvOpDecorate: {
        const auto target = inst.words[1];
        if (inst.num_words == 4) {
          const auto number = inst.words[3];
          switch (inst.words[2]) {
            case SpvDecorationDescriptorSet:
            case SpvDecorationBinding:
              SaveDescriptorInfo(target, SpvDecoration(inst.words[2]), number);
              break;
            default:
              break;
          }
        }
      } break;
      case SpvOpFunction:
        current_function_ = inst.words[2];
        break;
      case SpvOpFunctionEnd: {
        (*entry_points_)[entry_point_map_[current_function_]].descriptors() =
            uses_[current_function_];
        current_function_ = 0;
      } break;
      case SpvOpLoad:
        SaveReferenceIfDescriptor(inst.words[3]);
        break;
      case SpvOpStore:
        SaveReferenceIfDescriptor(inst.words[1]);
        break;
      case SpvOpAccessChain:
      case SpvOpInBoundsAccessChain:
      case SpvOpPtrAccessChain:
      case SpvOpInBoundsPtrAccessChain:
        SaveReferenceIfDescriptor(inst.words[3]);
        break;
      case SpvOpFunctionCall: {
        // For a function call, each operand is a single word.  The call
        // operands start at word 4.
        for (unsigned i = 4; i < inst.num_words; i++) {
          SaveReferenceIfDescriptor(inst.words[i]);
        }
      } break;
      case SpvOpImageTexelPointer:
        SaveReferenceIfDescriptor(inst.words[3]);
        break;
      case SpvOpAtomicLoad:
        SaveReferenceIfDescriptor(inst.words[3]);
        break;
      case SpvOpAtomicStore:
        SaveReferenceIfDescriptor(inst.words[1]);
        break;
      case SpvOpAtomicExchange:
      case SpvOpAtomicCompareExchange:
      case SpvOpAtomicCompareExchangeWeak:
      case SpvOpAtomicIIncrement:
      case SpvOpAtomicIDecrement:
      case SpvOpAtomicIAdd:
      case SpvOpAtomicISub:
      case SpvOpAtomicSMin:
      case SpvOpAtomicUMin:
      case SpvOpAtomicSMax:
      case SpvOpAtomicUMax:
      case SpvOpAtomicAnd:
      case SpvOpAtomicOr:
      case SpvOpAtomicXor:
      case SpvOpAtomicFlagTestAndSet:
        SaveReferenceIfDescriptor(inst.words[3]);
        break;
      case SpvOpAtomicFlagClear:
        SaveReferenceIfDescriptor(inst.words[1]);
        break;
      case SpvOpCopyObject:
        SaveReferenceIfDescriptor(inst.words[3]);
        break;
      case SpvOpCopyMemory:
        SaveReferenceIfDescriptor(inst.words[1]);
        SaveReferenceIfDescriptor(inst.words[2]);
        break;
      default:
        break;
    }
    return SPV_SUCCESS;
  }

  // Saves the fact about the descriptor set or binding information for
  // the given target id.
  void SaveDescriptorInfo(uint32_t target, SpvDecoration decoration,
                          uint32_t number) {
    auto& d = id_descriptor_map_[target];
    if (decoration == SpvDecorationDescriptorSet) {
      d.set = number;
    } else if (decoration == SpvDecorationBinding) {
      d.binding = number;
    }
  }

  // If the given Id is a direct or indirect reference to a variable with
  // a descriptor, then record this function's use of the descriptor.
  void SaveReferenceIfDescriptor(uint32_t id) {
    if (current_function_) {
      const auto& where = id_descriptor_map_.find(id);
      if (where != id_descriptor_map_.end()) {
        uses_[current_function_].insert(where->second);
      }
    }
 } 

 private:
  // The accumulated entry point information.
  std::vector<EntryPointInfo>* entry_points_;

  // Map the Id of an entry point to its index in entry_points_.
  std::unordered_map<uint32_t, uint32_t> entry_point_map_;

  // The Id of the current function. A function is current if we have seen its
  // OpFunction instruction but not its OpFunctionEnd instruction.  This is 0
  // when there is no current function.
  uint32_t current_function_;

  // Maps the Id of a function to the directly referenced descriptors.
  std::unordered_map<uint32_t, Descriptors> uses_;

  // Maps an Id to the descriptor decorated on it.
  std::unordered_map<uint32_t, Descriptor> id_descriptor_map_;
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
