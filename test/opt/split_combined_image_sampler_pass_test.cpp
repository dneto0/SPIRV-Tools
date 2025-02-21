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

#include "spirv-tools/optimizer.hpp"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

// using SplitCombinedImageSamplerPassTest = PassTest<::testing::Test>;

struct SplitCombinedImageSamplerPassTest : public PassTest<::testing::Test> {
  virtual void SetUp() override {
    SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
                          SPV_BINARY_TO_TEXT_OPTION_INDENT |
                          SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  }
};

// Valid original types (GLSL syntax), float sampled types
//   sampler2D
//   sampler2DShadow
//   sampler2DArray
//   sampler2DArrayShadow
//   sampler2DMS (not actually sampled...)
//   sampler2DMSArray (not actually sampled...)
//   sampler3D
//   samplerCube
//   samplerCubeShadow
//   samplerCubeArray
//   samplerCubeArrayShadow

#if 0
TEST_F(SplitCombinedImageSamplerPassTest, NoCombined_NoChange)
TEST_F(SplitCombinedImageSamplerPassTest, Combined_
TEST_F(SplitCombinedImageSamplerPassTest, CombinedBindingNumber
TEST_F(SplitCombinedImageSamplerPassTest, BindingNumbersOtherResources)
#endif

std::string PreambleStr() {
  return R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
               OpExecutionMode %1 LocalSize 1 1 1
)";
}
std::string MainStr() {
  return R"(       %void = OpTypeVoid
          %3 = OpTypeFunction %void
          %1 = OpFunction %void None %3
          %6 = OpLabel
               OpReturn
               OpFunctionEnd
)";
}
std::string NoCheck() { return "; CHECK-NOT: nothing to see"; }

TEST_F(SplitCombinedImageSamplerPassTest, NoCombined_NoChange) {
  const std::string kTest = PreambleStr() + MainStr();
  auto result = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest + NoCheck(), /* do_validation= */ true);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
  EXPECT_EQ(std::get<0>(result), kTest);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
