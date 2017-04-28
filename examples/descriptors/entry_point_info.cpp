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

#include "entry_point_info.h"

namespace spirv_example {

spv_result_t GetEntryPointInfo(const spv_const_context,
                               const uint32_t* words, size_t num_words,
                               std::vector<EntryPointInfo>* entry_points,
                               spv_diagnostic* diagnostic) {
  words = words;
  num_words = num_words;
  diagnostic = diagnostic;
  if (!entry_points) return SPV_ERROR_INVALID_POINTER;

  return SPV_SUCCESS;
}

}  // namespace spirv_example
