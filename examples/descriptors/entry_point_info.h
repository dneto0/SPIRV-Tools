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

#ifndef ENTRY_POINT_INFO_H_
#define ENTRY_POINT_INFO_H_

#include "spirv-tools/libspirv.h"

#include <set>
#include <string>
#include <vector>

namespace spirv_example {

// The interesting facts about a descriptor binding.
struct Descriptor {
  uint32_t set;      // The DescriptorSet decoration.
  uint32_t binding;  // The Binding decoration.
};
using Descriptors = std::set<Descriptor>;

inline bool operator<(const Descriptor& lhs, const Descriptor& rhs) {
  return lhs.set < rhs.set || (lhs.set == rhs.set && lhs.binding < lhs.binding);
}

inline bool operator==(const Descriptor& lhs, const Descriptor& rhs) {
  return lhs.set == rhs.set && lhs.binding == rhs.binding;
}


// Facts about a single entry point.
class EntryPointInfo {
 public:
  EntryPointInfo() {}
  EntryPointInfo(const std::string& _name, const Descriptors& descriptors = {})
      : name_(_name), descriptors_(descriptors) {}

  const std::string& name() const { return name_; }
 private:
  // The name of the entry point.
  std::string name_;
  // The set of descriptors for variables referenced by this entry point.
  Descriptors descriptors_;
};

inline bool operator==(const EntryPointInfo& lhs, const EntryPointInfo& rhs) {
  return lhs.name() == rhs.name();
}


// Builds a description the entry points in the valid module specified as
// |num_words| words at |words|.  The |entry_points| pointer must not be null.
// Returns SPV_SUCCESS on success.  On failure, populates the |diagnostic|
// argument, if that diagnostic is not null.
spv_result_t GetEntryPointInfo(const spv_const_context context,
                               const uint32_t* words, size_t num_words,
                               std::vector<EntryPointInfo>* entry_points,
                               spv_diagnostic* diagnostic);

}  // namespace spirv_example

#endif  // ENTRY_POINT_INFO_H_
