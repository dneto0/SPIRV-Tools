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

#include <array>
#include <iostream>
#include <ostream>

#include "spirv-tools/optimizer.hpp"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

struct TypeCase {
  const char* glsl_type;
  const char* image_type_decl;
};
std::ostream& operator<<(std::ostream& os, const TypeCase& tc) {
  os << tc.glsl_type;
  return os;
}

struct SplitCombinedImageSamplerPassTest : public PassTest<::testing::Test> {
  virtual void SetUp() override {
    SetTargetEnv(SPV_ENV_VULKAN_1_0);
    SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
                          SPV_BINARY_TO_TEXT_OPTION_INDENT |
                          SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  }
};

struct SplitCombinedImageSamplerPassTypeCaseTest
    : public PassTest<::testing::TestWithParam<TypeCase>> {
  virtual void SetUp() override {
    SetTargetEnv(SPV_ENV_VULKAN_1_0);
    SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
                          SPV_BINARY_TO_TEXT_OPTION_INDENT |
                          SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  }
};

std::vector<TypeCase> Cases() {
  return std::vector<TypeCase>{
      {"sampler2D", "OpTypeImage %float 2D 0 0 0 1 Unknown"},
      {"sampler2DShadow", "OpTypeImage %float 2D 1 0 0 1 Unknown"},
      {"sampler2DArray", "OpTypeImage %float 2D 0 1 0 1 Unknown"},
      {"sampler2DArrayShadow", "OpTypeImage %float 2D 1 1 0 1 Unknown"},
      {"sampler2DMS", "OpTypeImage %float 2D 0 0 1 1 Unknown"},
      {"sampler2DMSArray", "OpTypeImage %float 2D 0 1 1 1 Unknown"},
      {"sampler3D", "OpTypeImage %float 3D 0 0 0 1 Unknown"},
      {"samplerCube", "OpTypeImage %float Cube 0 0 0 1 Unknown"},
      {"samplerCubeShadow", "OpTypeImage %float Cube 1 0 0 1 Unknown"},
      {"samplerCubeArray", "OpTypeImage %float Cube 0 1 0 1 Unknown"},
      {"samplerCubeArrayShadow", "OpTypeImage %float Cube 1 1 0 1 Unknown"},
      {"isampler2D", "OpTypeImage %int 2D 0 0 0 1 Unknown"},
      {"isampler2DShadow", "OpTypeImage %int 2D 1 0 0 1 Unknown"},
      {"isampler2DArray", "OpTypeImage %int 2D 0 1 0 1 Unknown"},
      {"isampler2DArrayShadow", "OpTypeImage %int 2D 1 1 0 1 Unknown"},
      {"isampler2DMS", "OpTypeImage %int 2D 0 0 1 1 Unknown"},
      {"isampler2DMSArray", "OpTypeImage %int 2D 0 1 1 1 Unknown"},
      {"isampler3D", "OpTypeImage %int 3D 0 0 0 1 Unknown"},
      {"isamplerCube", "OpTypeImage %int Cube 0 0 0 1 Unknown"},
      {"isamplerCubeShadow", "OpTypeImage %int Cube 1 0 0 1 Unknown"},
      {"isamplerCubeArray", "OpTypeImage %int Cube 0 1 0 1 Unknown"},
      {"isamplerCubeArrayShadow", "OpTypeImage %int Cube 1 1 0 1 Unknown"},
      {"usampler2D", "OpTypeImage %uint 2D 0 0 0 1 Unknown"},
      {"usampler2DShadow", "OpTypeImage %uint 2D 1 0 0 1 Unknown"},
      {"usampler2DArray", "OpTypeImage %uint 2D 0 1 0 1 Unknown"},
      {"usampler2DArrayShadow", "OpTypeImage %uint 2D 1 1 0 1 Unknown"},
      {"usampler2DMS", "OpTypeImage %uint 2D 0 0 1 1 Unknown"},
      {"usampler2DMSArray", "OpTypeImage %uint 2D 0 1 1 1 Unknown"},
      {"usampler3D", "OpTypeImage %uint 3D 0 0 0 1 Unknown"},
      {"usamplerCube", "OpTypeImage %uint Cube 0 0 0 1 Unknown"},
      {"usamplerCubeShadow", "OpTypeImage %uint Cube 1 0 0 1 Unknown"},
      {"usamplerCubeArray", "OpTypeImage %uint Cube 0 1 0 1 Unknown"},
      {"usamplerCubeArrayShadow", "OpTypeImage %uint Cube 1 1 0 1 Unknown"},
  };
}

#if 0
TEST_F(SplitCombinedImageSamplerPassTest, NoCombined_NoChange)
TEST_F(SplitCombinedImageSamplerPassTest, Combined_
TEST_F(SplitCombinedImageSamplerPassTest, CombinedBindingNumber
TEST_F(SplitCombinedImageSamplerPassTest, BindingNumbersOtherResources)
#endif

std::string Preamble() {
  return R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpName %main "main"
               OpName %main_0 "main_0"
               OpName %voidfn "voidfn"
)";
}

std::string BasicTypes() {
  return R"(      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
       %void = OpTypeVoid
     %voidfn = OpTypeFunction %void
)";
}
std::string Main() {
  return R"(       %voidfn = OpTypeFunction %void
       %main = OpFunction %void None %voidfn
          %6 = OpLabel
               OpReturn
               OpFunctionEnd
)";
}
std::string NoCheck() { return "; CHECK-NOT: nothing to see"; }

std::string Decorations() {
  return R"(               OpName %im_ty "im_ty"
               OpName %s_ty "s_ty"
               OpName %ims_ty "ims_ty"
               OpName %ims_pty "ims_pty"
               OpDecorate %100 DescriptorSet 0
               OpDecorate %100 Binding 0
)";
}

std::string Decls(const std::string& image_type_decl) {
  return R"(       %s_ty = OpTypeSampler
      %im_ty = )" +
         image_type_decl + R"(
     %ims_ty = OpTypeSampledImage %im_ty
    %ims_pty = OpTypePointer UniformConstant %ims_ty
        %100 = OpVariable %ims_pty UniformConstant
)";
}

TEST_F(SplitCombinedImageSamplerPassTest, SamplerOnly_NoChange) {
  const std::string kTest = Preamble() +
                            R"(               OpDecorate %100 DescriptorSet 0
               OpDecorate %100 Binding 0
)" + BasicTypes() + R"(         %10 = OpTypeSampler
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
        %100 = OpVariable %_ptr_UniformConstant_10 UniformConstant
       %main = OpFunction %void None %voidfn
     %main_0 = OpLabel
          %6 = OpLoad %10 %100
               OpReturn
               OpFunctionEnd
)";

  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest + NoCheck(), /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithoutChange) << "status";
  EXPECT_EQ(disasm, kTest) << "disasm";
}

TEST_F(SplitCombinedImageSamplerPassTest, ImageOnly_NoChange) {
  const std::string kTest = Preamble() +
                            R"(               OpDecorate %100 DescriptorSet 0
               OpDecorate %100 Binding 0
)" + BasicTypes() + R"(         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
        %100 = OpVariable %_ptr_UniformConstant_10 UniformConstant
       %main = OpFunction %void None %voidfn
     %main_0 = OpLabel
          %6 = OpLoad %10 %100
               OpReturn
               OpFunctionEnd
)";

  SCOPED_TRACE("image only");
  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest + NoCheck(), /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithoutChange);
  EXPECT_EQ(disasm, kTest);
}

TEST_F(SplitCombinedImageSamplerPassTest, Combined_NoSampler_CreatedAtFront) {
  // No OpTypeSampler to begin with.
  const std::string kTest = Preamble() +
                            R"(               OpDecorate %100 DescriptorSet 0
               OpDecorate %100 Binding 0

     ; A sampler type is created and placed at the start of types.
     ; CHECK: OpDecorate %100 Binding 0
     ; CHECK-NEXT: %[[sampler_ty:\d+]] = OpTypeSampler
     ; CHECK-NEXT: OpTypeBool

               %bool = OpTypeBool ; location marker
)" + BasicTypes() + R"( %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
         %11 = OpTypeSampledImage %10
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11

        %100 = OpVariable %_ptr_UniformConstant_11 UniformConstant
       %main = OpFunction %void None %voidfn
     %main_0 = OpLabel
          %6 = OpLoad %11 %100
               OpReturn
               OpFunctionEnd
)";
  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest + NoCheck(), /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange);
}

TEST_F(SplitCombinedImageSamplerPassTest, Combined_Sampler_MovedToFront) {
  // No OpTypeSampler to begin with.
  const std::string kTest = Preamble() +
                            R"(               OpDecorate %100 DescriptorSet 0
               OpDecorate %100 Binding 0

     ; The sampler type is moved to the front.
     ; CHECK: OpDecorate %100 Binding 0
     ; CHECK-NEXT: %99 = OpTypeSampler
     ; CHECK-NEXT: OpTypeBool
     ; CHECK-NOT: OpTypeSampler
     ; CHECK: OpFunction %void

               %bool = OpTypeBool ; location marker
)" + BasicTypes() +
                            R"(%10 = OpTypeImage %float 2D 0 0 0 1 Unknown
         %11 = OpTypeSampledImage %10
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11

        %99 = OpTypeSampler

        %100 = OpVariable %_ptr_UniformConstant_11 UniformConstant
       %main = OpFunction %void None %voidfn
     %main_0 = OpLabel
          %6 = OpLoad %11 %100
               OpReturn
               OpFunctionEnd
)";
  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest + NoCheck(), /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange);
}

TEST_P(SplitCombinedImageSamplerPassTypeCaseTest, Combined_Load) {
  // No OpTypeSampler to begin with.
  const std::string kTest = Preamble() +
                            R"(               OpDecorate %100 DescriptorSet 0
               OpDecorate %100 Binding 0

     ; CHECK: %[[sampler_ty:\d+]] = OpTypeSampler

               %bool = OpTypeBool ; location marker
)" + BasicTypes() +
                            "%10 = " + GetParam().image_type_decl + R"(
         %11 = OpTypeSampledImage %10
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11

        %100 = OpVariable %_ptr_UniformConstant_11 UniformConstant
       %main = OpFunction %void None %voidfn
     %main_0 = OpLabel
          %6 = OpLoad %11 %100
               OpReturn
               OpFunctionEnd
)";
  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest + NoCheck(), /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange);
}

INSTANTIATE_TEST_SUITE_P(AllCombinedTypes,
                         SplitCombinedImageSamplerPassTypeCaseTest,
                         ::testing::ValuesIn(Cases()));

}  // namespace
}  // namespace opt
}  // namespace spvtools
