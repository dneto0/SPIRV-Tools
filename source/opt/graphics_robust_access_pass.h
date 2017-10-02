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

#ifndef LIBSPIRV_OPT_GRAPHICS_ROBUST_ACCESS_PASS_H_
#define LIBSPIRV_OPT_GRAPHICS_ROBUST_ACCESS_PASS_H_

#include <unordered_map>
#include <map>

#include "diagnostic.h"

#include "def_use_manager.h"
#include "module.h"
#include "pass.h"


namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class GraphicsRobustAccessPass : public Pass {
 public:
  GraphicsRobustAccessPass();
  const char* name() const override { return "graphics-robust-access"; }
  Status Process(ir::Module*) override;

 private:
  // Records failure for the current module, and returns a stream
  // that can be used to provide user error information to the message
  // consumer.
  libspirv::DiagnosticStream Fail();

  // Transform the current module, if possible. Failure and modification
  // status is recorded in the |_| member. On failure, error information is
  // posted to the message consumer.  The return value has no significance.
  spv_result_t ProcessCurrentModule();

  // Process the given function.  Updates the state value |_|.  Returns true
  // if the module was modified.
  bool ProcessAFunction(ir::Function*);

  // Clamps indices in address calculation instruction referenced by the
  // instruction iterator.  Inserts instructions before the given instruction,
  // and updates the given iterator.  Updates _.modified as required.
  void ClampIndicesForAccessChain(ir::BasicBlock::iterator* inst_iter);

  // Returns the id of the GLSL.std.450 extended instruction set.  Creates it if it
  // does not yet exist.  Updates _.modified as required.
  uint32_t GetGlslInsts();
  // Returns the id of the unsigend type of the given bit width.  Creates a
  // type definition instruction if needed, and updates internal state as
  // required.
  uint32_t GetUintType(uint32_t width);
  // Returns the Id of a constant with the given value using the given type Id.
  // Creates a constant instruction if needed, and updates internal state as
  // required.
  uint32_t GetUintValue(uint32_t type_id, uint64_t value);

  // Get the instruction that defines the given id.
  ir::Instruction* GetDef(uint32_t id) {
    return _.def_use_mgr->GetDef(id);
  }

  // Returns the numeric valid for an OpConstant instruction for an unsigned
  // integer type of up to 64 bits.
  uint64_t GetUintValueFromConstant(const ir::Instruction& inst);

  // Returns a new instruction that is a call to the Umax extended instruction
  // with the two given operands.  The must both be of the same unsigned
  // integer type.
  std::unique_ptr<ir::Instruction> MakeUmaxInst(uint32_t id0, uint32_t id1);

  // Record the width of each signed or unsigned integer type, by id.  Only
  // handles widths up to 64 bits.
  void LoadIntTypeWidths();
  // Record the id of all unsigned integer constants up to 64 bits wide.  This
  // is only valid to call if unsigned int types have been recorded.
  void LoadUintValues();

  // A pair representing the id of an unsigned integer type, and a value in that
  // type's range.
  using TypeValue = std::pair<uint32_t, uint64_t>;

  // State required for the current state.
  struct PerModuleState {
    PerModuleState(ir::Module* m)
        : module(m),
          def_use_mgr(new analysis::DefUseManager(nullptr, m)),
          next_id(m ? m->IdBound() : 0) {}

    // The module currently being processed.
    ir::Module* module;

    // Definition-and-use tables for the current module.
    std::unique_ptr<analysis::DefUseManager> def_use_mgr;

    // This pass modified the module.
    bool modified = false;
    // True if there is an error processing the current module, e.g. if
    // preconditions are not met.
    bool failed = false;

    // The next id to use.
    uint32_t next_id = 0;
    // The id of the GLSL.std.450 extended instruction set.  Zero if it does
    // not exist.
    uint32_t glsl_insts_id = 0;
    // Maps a bit width to the Id of the unsigned integer type of that width.
    // Only handles widts up to 64 bits.
    std::unordered_map<uint32_t, uint32_t> uint_type = {};
    // Maps a type id of an unsigned integer type to its width.
    // Only handles widths up to 64 bits.
    std::unordered_map<uint32_t, uint32_t> width_of_uint_type = {};
    // Maps a type id of an signed integer type to its width.
    // Only handles widths up to 64 bits.
    std::unordered_map<uint32_t, uint32_t> width_of_int_type = {};
    // Maps an unsigned integer value of a given type Id to the Id of a constant
    // with that value.  The pair is specified as (type id, value).
    // Only handles widts up to 64 bits.
    std::map<TypeValue, uint32_t> uint_value = {};
  } _;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_GRAPHICS_ROBUST_ACCESS_PASS_H_
