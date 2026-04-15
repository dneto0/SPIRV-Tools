// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#ifndef SOURCE_SPIRV_ENDIAN_H_
#define SOURCE_SPIRV_ENDIAN_H_

#include "spirv-tools/libspirv.h"

// Converts a word in the specified endianness to the host native endianness.
uint32_t spvFixWord(const uint32_t word, const spv_endianness_t endianness);

// Converts a word from one endianness to the other.
inline uint32_t spvFlipWord(const uint32_t word) {
  return ((word >> 24u) & 0xffu) | ((word >> 8u) & 0xff00u) |
         ((word << 8u) & 0xff0000u) | ((word << 24u) & 0xff000000u);
}

// Converts a pair of words in the specified endianness to the host native
// endianness.
uint64_t spvFixDoubleWord(const uint32_t low, const uint32_t high,
                          const spv_endianness_t endianness);

// Gets the endianness of the SPIR-V module given in the binary parameter.
// Returns SPV_ENDIANNESS_UNKNOWN if the SPIR-V magic number is invalid,
// otherwise writes the determined endianness into *endian.
spv_result_t spvBinaryEndianness(const spv_const_binary binary,
                                 spv_endianness_t* endian);

// Returns true if the given endianness matches the host's native endianness.
bool spvIsHostEndian(spv_endianness_t endian);

#endif  // SOURCE_SPIRV_ENDIAN_H_
