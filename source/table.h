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

#ifndef SOURCE_TABLE_H_
#define SOURCE_TABLE_H_

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

typedef struct spv_opcode_desc_t {
  const char* name;
  const spv::Op opcode;
  const uint32_t numAliases;
  const char** aliases;
  const uint32_t numCapabilities;
  const spv::Capability* capabilities;
  // operandTypes[0..numTypes-1] describe logical operands for the instruction.
  // The operand types include result id and result-type id, followed by
  // the types of arguments.
  const uint16_t numTypes;
  spv_operand_type_t operandTypes[16];  // TODO: Smaller/larger?
  const bool hasResult;  // Does the instruction have a result ID operand?
  const bool hasType;    // Does the instruction have a type ID operand?
  // A set of extensions that enable this feature. If empty then this operand
  // value is in core and its availability is subject to minVersion. The
  // assembler, binary parser, and disassembler ignore this rule, so you can
  // freely process invalid modules.
  //
  const uint32_t numExtensions;
  const spvtools::Extension* extensions;
  // Minimal core SPIR-V version required for this feature, if without
  // extensions. ~0u means reserved for future use. ~0u and non-empty extension
  // lists means only available in extensions.
  const uint32_t minVersion;
  const uint32_t lastVersion;
} spv_opcode_desc_t;

typedef struct spv_operand_desc_t {
  const char* name;
  const uint32_t value;
  const uint32_t numAliases;
  const char** aliases;
  const uint32_t numCapabilities;
  const spv::Capability* capabilities;
  // A set of extensions that enable this feature. If empty then this operand
  // value is in core and its availability is subject to minVersion. The
  // assembler, binary parser, and disassembler ignore this rule, so you can
  // freely process invalid modules.
  const uint32_t numExtensions;
  const spvtools::Extension* extensions;
  const spv_operand_type_t operandTypes[16];  // TODO: Smaller/larger?
  // Minimal core SPIR-V version required for this feature, if without
  // extensions. ~0u means reserved for future use. ~0u and non-empty extension
  // lists means only available in extensions.
  const uint32_t minVersion;
  const uint32_t lastVersion;
} spv_operand_desc_t;

typedef struct spv_operand_desc_group_t {
  const spv_operand_type_t type;
  const uint32_t count;
  const spv_operand_desc_t* entries;
} spv_operand_desc_group_t;

typedef struct spv_ext_inst_desc_t {
  const char* name;
  const uint32_t ext_inst;
  const uint32_t numCapabilities;
  const spv::Capability* capabilities;
  const spv_operand_type_t operandTypes[40];  // vksp needs at least 40
} spv_ext_inst_desc_t;

typedef struct spv_ext_inst_group_t {
  const spv_ext_inst_type_t type;
  const uint32_t count;
  const spv_ext_inst_desc_t* entries;
} spv_ext_inst_group_t;

typedef struct spv_opcode_table_t {
  const uint32_t count;
  const spv_opcode_desc_t* entries;
} spv_opcode_table_t;

typedef struct spv_operand_table_t {
  const uint32_t count;
  const spv_operand_desc_group_t* types;
} spv_operand_table_t;

typedef struct spv_ext_inst_table_t {
  const uint32_t count;
  const spv_ext_inst_group_t* groups;
} spv_ext_inst_table_t;

typedef const spv_opcode_desc_t* spv_opcode_desc;
typedef const spv_operand_desc_t* spv_operand_desc;
typedef const spv_ext_inst_desc_t* spv_ext_inst_desc;

typedef const spv_opcode_table_t* spv_opcode_table;
typedef const spv_operand_table_t* spv_operand_table;
typedef const spv_ext_inst_table_t* spv_ext_inst_table;

struct spv_context_t {
  const spv_target_env target_env;
  const spv_operand_table operand_table;
  const spv_ext_inst_table ext_inst_table;
  spvtools::MessageConsumer consumer;
};

namespace spvtools {

// Sets the message consumer to |consumer| in the given |context|. The original
// message consumer will be overwritten.
void SetContextMessageConsumer(spv_context context, MessageConsumer consumer);
}  // namespace spvtools

// Populates *table with entries for env.
spv_result_t spvOperandTableGet(spv_operand_table* table, spv_target_env env);

// Populates *table with entries for env.
spv_result_t spvExtInstTableGet(spv_ext_inst_table* table, spv_target_env env);

#endif  // SOURCE_TABLE_H_
