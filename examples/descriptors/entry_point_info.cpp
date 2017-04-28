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
#include <utility>
#include <vector>

#include "spirv/1.1/spirv.h"
#include "spirv-tools/libspirv.h"

using spirv_example::EntryPointInfo;

namespace {

// Returns a string for the literal string logical operand starting at the
// given word.  assumes it has a terminating null as required.
std::string StringFromWords(const uint32_t* words) {
  return reinterpret_cast<const char*>(words);
}

// holds accumulated information about all entry points in the module.
class Collector {
 public:
  explicit Collector(std::vector<EntryPointInfo>* entry_points)
      : entry_points_(entry_points) {
    entry_points_->clear();
  }

  spv_result_t HandleInstruction(const spv_parsed_instruction_t& inst) {
    if (inst.opcode == SpvOpEntryPoint) {
      entry_points_->push_back(EntryPointInfo{StringFromWords(inst.words + 3)});
    }
    return SPV_SUCCESS;
  }

 private:
  // The accumulated entry point information.
  std::vector<EntryPointInfo>* entry_points_;
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
