// Copyright (c) 2025 The Khronos Group Inc.
// Copyright (c) 2025 Google LLC
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

#ifndef SOURCE_TABLE2_H_
#define SOURCE_TABLE2_H_

#include "source/extensions.h"
#include "source/latest_version_spirv_header.h"
#include "source/util/index_range.h"
#include "spirv-tools/libspirv.hpp"

// Define the static tables that describe the grammatical structure
// of SPIR-V instructions and their operands. These tables are populated
// by reading the grammar files from SPIRV-Headers.
//
// Most clients access these tables indirectly via an spv_context_t object.
//
// It should be very fast, and require no memory allocations, to create
// an spv_context_t object.
// It should be very fast for the system loader to load (and possibly relocate)
// the tables.  In particular, there should be very few global symbols with
// independent addresses. Prefer a very few large tables of items rather than
// dozens or hundreds of global symbols.
//
// The overall structure among containers (i.e. skipping scalar data members)
// is as follows:
//
//    An spv_context_t:
//      - points to spv_opcode_table_t = array of spv_opcode_desc_t
//      - points to spv_operand_table_t = array of spv_operand_desc_group_t
//      - points to spv_ext_inst_table_t  = array of spv_ext_inst_group_t
//
//    An spv_opcode_desc_t has:
//      - a name string
//      - array of alias strings
//      - array of spv::Capability      (an enum)
//      - array of spv_operand_type_t   (an enum)
//      - array of spvtools::Extension  (an enum)
//
//    An spv_operand_desc_group_t has:
//      - array of spv_operand_desc_t:
//
//    An spv_operand_desc_t has:
//      - a name string
//      - array of alias strings
//      - array of spv::Capability
//      - array of spvtools::Extension
//      - array of spv_operand_type_t
//
//    An spv_ext_inst_group_t has:
//      - array of spv_ext_inst_desc_t
//
//    An spv_ext_inst_desc_t has:
//      - a name string
//      - array of spv::Capability
//      - array of spv_operand_type_t
//
// The arrays are represented by spans into a global static array, with one
// array for each of:
//      - null-terminated strings, for names
//      - arrays of null-terminated strings, for alias lists
//      - spv_operand_type_t
//      - spv::Capability
//      - spvtools::Extension
//
// Note: Currently alias lists never have more than one element.

namespace spvtools {

using IndexRange = utils::IndexRange<uint32_t, uint32_t>;

constexpr inline IndexRange IR(uint32_t first, uint32_t count) {
  return IndexRange{first, count};
}

struct NameValue {
  // Location of the null-terminated name in the global string table kStrings.
  IndexRange name;
  // Enum value in the binary format.
  uint32_t value;
};

// Describes a SPIR-V operand.
struct OperandDesc {
  uint32_t value;

  IndexRange operands_range;      // Indexes kOperandSpans
  IndexRange name_range;          // Indexes kStrings
  IndexRange aliases_range;       // Indexes kAliasSpans
  IndexRange capabilities_range;  // Indexes kCapabilitySpans
  // A set of extensions that enable this feature. If empty then this operand
  // value is in core and its availability is subject to minVersion. The
  // assembler, binary parser, and disassembler ignore this rule, so you can
  // freely process invalid modules.
  IndexRange extensions_range;  // Indexes kExtensionSpans
  // Minimal core SPIR-V version required for this feature, if without
  // extensions. ~0u means reserved for future use. ~0u and non-empty
  // extension lists means only available in extensions.
  uint32_t minVersion;
  uint32_t lastVersion;
  utils::Span<spv_operand_type_t> operands() const;
  utils::Span<const char> name() const;
  utils::Span<IndexRange> aliases() const;
  utils::Span<spv::Capability> capabilities() const;
  utils::Span<spvtools::Extension> extensions() const;

  OperandDesc(const OperandDesc&) = delete;
  OperandDesc(OperandDesc&&) = delete;
};

// Describes an Instruction
struct InstructionDesc {
  const spv::Op opcode;
  const bool hasResult;
  const bool hasType;

  const IndexRange operands_range;      // Indexes kOperandSpans
  const IndexRange name_range;          // Indexes kStrings
  const IndexRange aliases_range;       // Indexes kAliasSpans
  const IndexRange capabilities_range;  // Indexes kCapbilitySpans
  // A set of extensions that enable this feature. If empty then this operand
  // value is in core and its availability is subject to minVersion. The
  // assembler, binary parser, and disassembler ignore this rule, so you can
  // freely process invalid modules.
  const IndexRange extensions_range;  // Indexes kExtensionSpans
  // Minimal core SPIR-V version required for this feature, if without
  // extensions. ~0u means reserved for future use. ~0u and non-empty
  // extension lists means only available in extensions.
  uint32_t minVersion;
  uint32_t lastVersion;
  // Returns the span of elements in the global grammar tables corresponding
  // to the privately-stored index ranges
  utils::Span<spv_operand_type_t> operands() const;
  utils::Span<const char> name() const;
  utils::Span<IndexRange> aliases() const;
  utils::Span<spv::Capability> capabilities() const;
  utils::Span<spvtools::Extension> extensions() const;

  InstructionDesc(const InstructionDesc&) = delete;
  InstructionDesc(InstructionDesc&&) = delete;
};

// Returns a pointer to the null-terminated C-style string. Assumes the given
// index range is valid.
const char* getChars(IndexRange);

spv_result_t LookupOperand(spv_operand_type_t type, const char* name,
                           size_t name_len, OperandDesc* desc);
spv_result_t LookupOperand(spv_operand_type_t type, uint32_t operand,
                           OperandDesc* desc);

}  // namespace spvtools
#endif  // SOURCE_TABLE2_H_
