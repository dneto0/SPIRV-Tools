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

#include "gmock/gmock.h"

#include <sstream>
#include <vector>

#include "spirv-tools/libspirv.hpp"

#include "entry_point_info.h"

namespace {

using spirv_example::GetEntryPointInfo;
using spirv_example::EntryPointInfo;
using testing::Eq;

using Infos = std::vector<EntryPointInfo>;
using Words = std::vector<uint32_t>;

// A scoped SPIRV-Tools context object.
class Context {
 public:
  Context() : context(spvContextCreate(SPV_ENV_UNIVERSAL_1_1)) {}
  explicit Context(spv_target_env env) : context(spvContextCreate(env)) {}
  ~Context() { spvContextDestroy(context); }
  operator spv_context() { return context; }
  spv_context context;
};

// Assembles a SPIR-V module from a string.  Returns a vector of words.
// Assumes the assembly is valid.
std::vector<uint32_t> Assemble(std::string source) {
  std::vector<uint32_t> result;
  spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_1);
  std::ostringstream errs;
  tools.SetMessageConsumer([&errs](spv_message_level_t, const char*,
                                   const spv_position_t& pos,
                                   const char* message) {
    errs << pos.line << ":" << pos.column << ": " << message << "\n";
  });
  tools.Assemble(source, &result);
  EXPECT_GE(result.size(), 5u) << errs.str();
  return result;
}

TEST(EntryPointInfo, NullEntryPointsReturnsError) {
  EXPECT_EQ(SPV_ERROR_INVALID_POINTER,
            GetEntryPointInfo(Context(), nullptr, 0, nullptr, nullptr));
}

TEST(EntryPointInfo, NullBinaryReturnsError) {
  Infos infos;
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            GetEntryPointInfo(Context(), nullptr, 0, &infos, nullptr));
}

TEST(EntryPointInfo, BadBinaryReturnsError) {
  Infos infos;
  Words binary{1, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            GetEntryPointInfo(Context(), binary.data(), binary.size(), &infos,
                              nullptr));
}

TEST(EntryPointInfo, NoEntryPointsReturnsEmptyVector) {
  Infos infos;
  auto binary = Assemble(R"(
    OpCapability Addresses
    OpCapability Linkage
    OpCapability Kernel
    OpMemoryModel Physical32 OpenCL
  )");
  EXPECT_EQ(SPV_SUCCESS,
            GetEntryPointInfo(Context(), binary.data(), binary.size(), &infos,
                              nullptr));
  EXPECT_THAT(infos, Eq(Infos{}));
}

}  // anonymous namespace
