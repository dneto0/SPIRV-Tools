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

#include "bin_to_descriptors_str.h"

#include <vector>

#include "entry_point_info.h"

spv_result_t BinaryToDescriptorsStr(const spv_const_context context,
                                    const uint32_t* words, size_t num_words,
                                    std::ostream* out,
                                    spv_diagnostic* diagnostic) {
  std::vector<spirv_example::EntryPointInfo> entry_points;
  auto status = spirv_example::GetEntryPointInfo(context, words, num_words,
                                                 &entry_points, diagnostic);
  if (status == SPV_SUCCESS) {
    for (auto& entry_point : entry_points) {
      *out << entry_point.name() << std::endl;
    }
  }
  return SPV_SUCCESS;
}
