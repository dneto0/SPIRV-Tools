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

struct SplitCombinedImageSamplerPassTest : public PassTest<::testing::Test> {
  virtual void SetUp() override {
    SetTargetEnv(SPV_ENV_VULKAN_1_0);
    SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
                          SPV_BINARY_TO_TEXT_OPTION_INDENT |
                          SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  }
};

struct TypeCase {
  const char* glsl_type;
  const char* image_type_decl;
};
std::ostream& operator<<(std::ostream& os, const TypeCase& tc) {
  os << tc.glsl_type;
  return os;
}

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

std::vector<TypeCase> ImageTypeCases() {
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

std::string Preamble(const std::string shader_interface = "") {
  return R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main")" +
         shader_interface + R"(
               OpExecutionMode %main LocalSize 1 1 1
               OpName %main "main"
               OpName %main_0 "main_0"
               OpName %voidfn "voidfn"
)";
}

std::string PreambleFragment(const std::string shader_interface = "") {
  return R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main")" +
         shader_interface + R"(
               OpExecutionMode %main OriginUpperLeft
               OpName %main "main"
               OpName %main_0 "main_0"
               OpName %voidfn "voidfn"
)";
}

std::string BasicTypes() {
  return R"(      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
    %float_0 = OpConstant %float 0
    %v2float = OpTypeVector %float 2
    %v3float = OpTypeVector %float 3
    %v4float = OpTypeVector %float 4
         %13 = OpConstantNull %v2float
         %14 = OpConstantNull %v3float
         %15 = OpConstantNull %v4float
       %void = OpTypeVoid
     %voidfn = OpTypeFunction %void
)";
}
std::string Main() {
  return R"(
       %main = OpFunction %void None %voidfn
     %main_0 = OpLabel
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

TEST_F(SplitCombinedImageSamplerPassTest, PtrSampledImageOnly_DeletesPtrType) {
  const std::string kTest = Preamble() + BasicTypes() + R"(
  ; CHECK: OpCapability Shader
  ; CHECK-NOT: OpTypePointer UniformConstant
  ; CHECK: OpFunction %void
        %100 = OpTypeImage %float 2D 0 0 0 1 Unknown
        %101 = OpTypeSampledImage %100
        %102 = OpTypePointer UniformConstant %101
       %main = OpFunction %void None %voidfn
     %main_0 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest + NoCheck(), /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange) << "status";
}

TEST_F(SplitCombinedImageSamplerPassTest,
       PtrArraySampledImageOnly_DeletesPtrType) {
  const std::string kTest = Preamble() + BasicTypes() + R"(
  ; CHECK: OpCapability Shader
  ; CHECK-NOT: OpTypePointer UniformConstant
  ; CHECK: OpFunction %void
        %100 = OpTypeImage %float 2D 0 0 0 1 Unknown
        %101 = OpTypeSampledImage %100
     %uint_1 = OpConstant %uint 1
        %103 = OpTypeArray %101 %uint_1
        %104 = OpTypePointer UniformConstant %103
       %main = OpFunction %void None %voidfn
     %main_0 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest + NoCheck(), /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange) << "status";
}

TEST_F(SplitCombinedImageSamplerPassTest,
       PtrRtArraySampledImageOnly_DeletesPtrType) {
  const std::string kTest = Preamble() + BasicTypes() + R"(
  ; CHECK: OpCapability Shader
  ; CHECK-NOT: OpTypePointer UniformConstant
  ; CHECK: OpFunction %void
        %100 = OpTypeImage %float 2D 0 0 0 1 Unknown
        %101 = OpTypeSampledImage %100
        %103 = OpTypeRuntimeArray %101
        %104 = OpTypePointer UniformConstant %103
       %main = OpFunction %void None %voidfn
     %main_0 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest + NoCheck(), /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange) << "status";
}

TEST_F(SplitCombinedImageSamplerPassTest,
       Combined_NoSampler_CreatedBeforeSampledImage) {
  // No OpTypeSampler to begin with.
  const std::string kTest = Preamble() +
                            R"(               OpDecorate %100 DescriptorSet 0
               OpDecorate %100 Binding 0

     ; A sampler type is created and placed at the start of types.
     ; CHECK: OpDecorate %{{\d+}} Binding 0
     ; CHECK: OpDecorate %{{\d+}} Binding 0
     ; CHECK-NOT: TypeSampledImage
     ; CHECK: TypeSampler
     ; CHECK: TypeSampledImage

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
      kTest, /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange) << disasm;
}

TEST_P(SplitCombinedImageSamplerPassTypeCaseTest, Combined_RemapLoad) {
  const std::string kTest = Preamble() +
                            R"(
               OpName %combined "combined"
               OpDecorate %100 DescriptorSet 0
               OpDecorate %100 Binding 0

     ; CHECK: OpName
     ; CHECK-NOT: OpDecorate %100
     ; CHECK: OpDecorate %[[image_var:\d+]] DescriptorSet 0
     ; CHECK: OpDecorate %[[sampler_var:\d+]] DescriptorSet 0
     ; CHECK: OpDecorate %[[image_var]] Binding 0
     ; CHECK: OpDecorate %[[sampler_var]] Binding 0

     ; CHECK: %10 = OpTypeImage %
     ; CHECK: %[[image_ptr_ty:\w+]] = OpTypePointer UniformConstant %10
     ; CHECK: %[[sampler_ty:\d+]] = OpTypeSampler
     ; CHECK: %[[sampler_ptr_ty:\w+]] = OpTypePointer UniformConstant %[[sampler_ty]]

     ; The combined image variable is replaced by an image variable and a sampler variable.

     ; CHECK-NOT: %100 = OpVariable
     ; CHECK-DAG: %[[sampler_var]] = OpVariable %[[sampler_ptr_ty]] UniformConstant
     ; CHECK-DAG: %[[image_var]] = OpVariable %[[image_ptr_ty]] UniformConstant
     ; CHECK: = OpFunction

     ; The load of the combined image+sampler is replaced by a two loads, then
     ; a combination operation.
     ; CHECK: %[[im:\d+]] = OpLoad %10 %[[image_var]]
     ; CHECK: %[[s:\d+]] = OpLoad %[[sampler_ty]] %[[sampler_var]]
     ; CHECK: %combined = OpSampledImage %11 %[[im]] %[[s]]

               %bool = OpTypeBool ; location marker
)" + BasicTypes() +
                            " %10 = " + GetParam().image_type_decl + R"(
         %11 = OpTypeSampledImage %10
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11

        %100 = OpVariable %_ptr_UniformConstant_11 UniformConstant
       %main = OpFunction %void None %voidfn
     %main_0 = OpLabel
   %combined = OpLoad %11 %100

     ; Uses of the combined image sampler are preserved.
     ; CHECK: OpCopyObject %11 %combined

          %7 = OpCopyObject %11 %combined
               OpReturn
               OpFunctionEnd
)";
  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest, /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange) << disasm;
}

INSTANTIATE_TEST_SUITE_P(AllCombinedTypes,
                         SplitCombinedImageSamplerPassTypeCaseTest,
                         ::testing::ValuesIn(ImageTypeCases()));

// Remap entry point

struct EntryPointRemapCase {
  const spv_target_env environment = SPV_ENV_VULKAN_1_0;
  const char* initial_interface = "";
  const char* expected_interface = nullptr;
};

std::ostream& operator<<(std::ostream& os, const EntryPointRemapCase& eprc) {
  os << "(env " << spvLogStringForEnv(eprc.environment) << ", init "
     << eprc.initial_interface << " -> expect " << eprc.expected_interface
     << ")";
  return os;
}

struct SplitCombinedImageSamplerPassEntryPointRemapTest
    : public PassTest<::testing::TestWithParam<EntryPointRemapCase>> {
  virtual void SetUp() override {
    SetTargetEnv(GetParam().environment);
    SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
                          SPV_BINARY_TO_TEXT_OPTION_INDENT |
                          SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  }
};

std::vector<EntryPointRemapCase> EntryPointInterfaceCases() {
  return std::vector<EntryPointRemapCase>{
      {SPV_ENV_VULKAN_1_0, " %in_var %out_var", " %in_var %out_var"},
      {SPV_ENV_VULKAN_1_4, " %combined_var",
       " %[[image_var:\\d+]] %[[sampler_var:\\d+]]"},
      {SPV_ENV_VULKAN_1_4, " %combined_var %in_var %out_var",
       " %[[image_var:\\d+]] %in_var %out_var %[[sampler_var:\\d+]]"},
      {SPV_ENV_VULKAN_1_4, " %in_var %combined_var %out_var",
       " %in_var %[[image_var:\\d+]] %out_var %[[sampler_var:\\d+]]"},
      {SPV_ENV_VULKAN_1_4, " %in_var %out_var %combined_var",
       " %in_var %out_var %[[image_var:\\d+]] %[[sampler_var:\\d+]]"},
  };
};

TEST_P(SplitCombinedImageSamplerPassEntryPointRemapTest,
       EntryPoint_Combined_UsedInShader) {
  const bool combined_var_in_interface =
      std::string(GetParam().initial_interface).find("%combined_var") !=
      std::string::npos;
  // If the combined var is listed in the entry point, then the entry point
  // interface will give the pattern match definition of the sampler var ID.
  // Otherwise it's defined at the assignment.
  const std::string sampler_var_def =
      combined_var_in_interface ? "%[[sampler_var]]" : "%[[sampler_var:\\d+]]";
  const std::string image_var_def =
      combined_var_in_interface ? "%[[image_var]]" : "%[[image_var:\\d+]]";
  const std::string kTest = PreambleFragment(GetParam().initial_interface) +
                            R"(
               OpName %combined "combined"
               OpName %combined_var "combined_var"
               OpName %in_var "in_var"
               OpName %out_var "out_var"
               OpDecorate %combined_var DescriptorSet 0
               OpDecorate %combined_var Binding 0
               OpDecorate %in_var BuiltIn FragCoord
               OpDecorate %out_var Location 0

; CHECK: OpEntryPoint Fragment %main "main")" +
                            GetParam().expected_interface + R"(
; These clauses ensure the expected interface is the whole interface.
; CHECK-NOT: %{{\d+}}
; CHECK-NOT: %in_var
; CHECK-NOT: %out_var
; CHECK-NOT: %combined_var
; CHECK: OpExecutionMode %main OriginUpperLeft

     ; Check the var names, tracing up through the types.
     ; CHECK: %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
     ; CHECK: %[[image_ptr_ty:\w+]] = OpTypePointer UniformConstant %10
     ; CHECK: %[[sampler_ty:\d+]] = OpTypeSampler
     ; CHECK: %[[sampler_ptr_ty:\w+]] = OpTypePointer UniformConstant %[[sampler_ty]]
     ; The combined image variable is replaced by an image variable and a sampler variable.
     ; CHECK-DAG: )" + sampler_var_def +
                            R"( = OpVariable %[[sampler_ptr_ty]] UniformConstant
     ; CHECK-DAG: )" + image_var_def +
                            R"( = OpVariable %[[image_ptr_ty]] UniformConstant
     ; CHECK: = OpFunction

               %bool = OpTypeBool
)" + BasicTypes() + R"(         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
         %11 = OpTypeSampledImage %10
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11
     %in_ptr_v4f = OpTypePointer Input %v4float
     %in_var = OpVariable %in_ptr_v4f Input
    %out_ptr_v4f = OpTypePointer Output %v4float
    %out_var = OpVariable %out_ptr_v4f Output

%combined_var = OpVariable %_ptr_UniformConstant_11 UniformConstant
       %main = OpFunction %void None %voidfn
       ;CHECK:  %main_0 = OpLabel
       ;CHECK: OpLoad

     %main_0 = OpLabel
   %combined = OpLoad %11 %combined_var
               OpReturn
               OpFunctionEnd
)";
  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest, /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange) << disasm;
}

TEST_P(SplitCombinedImageSamplerPassEntryPointRemapTest,
       EntryPoint_Combined_UsedOnlyInEntryPointInstruction) {
  // If the combined var is in the interface, that is enough to trigger
  // its replacement. Otherwise the entry point interface is untouched
  // when the combined var is not otherwise used.
  const bool combined_var_in_interface =
      std::string(GetParam().initial_interface).find("%combined_var") !=
      std::string::npos;
  if (combined_var_in_interface) {
    const std::string kTest = PreambleFragment(GetParam().initial_interface) +
                              R"(
                 OpName %combined_var "combined_var"
                 OpName %in_var "in_var"
                 OpName %out_var "out_var"
                 OpDecorate %combined_var DescriptorSet 0
                 OpDecorate %combined_var Binding 0
                 OpDecorate %in_var BuiltIn FragCoord
                 OpDecorate %out_var Location 0

  ; CHECK: OpEntryPoint Fragment %main "main")" +
                              GetParam().expected_interface + R"(
  ; These clauses ensure the expected interface is the whole interface.
  ; CHECK-NOT: %{{\d+}}
  ; CHECK-NOT: %in_var
  ; CHECK-NOT: %out_var
  ; CHECK-NOT: %combined_var
  ; CHECK: OpExecutionMode %main OriginUpperLeft

                 %bool = OpTypeBool
  )" + BasicTypes() + R"(         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
           %11 = OpTypeSampledImage %10
  %_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11
       %in_ptr_v4f = OpTypePointer Input %v4float
       %in_var = OpVariable %in_ptr_v4f Input
      %out_ptr_v4f = OpTypePointer Output %v4float
      %out_var = OpVariable %out_ptr_v4f Output

  ; %combined_var is not used!
  %combined_var = OpVariable %_ptr_UniformConstant_11 UniformConstant
         %main = OpFunction %void None %voidfn
       %main_0 = OpLabel
                 OpReturn
                 OpFunctionEnd
  )";
    auto [disasm, status] =
        SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
            kTest, /* do_validation= */ true);
    EXPECT_EQ(status, Pass::Status::SuccessWithChange) << disasm;
  }
}

TEST_P(SplitCombinedImageSamplerPassEntryPointRemapTest,
       EntryPoint_Combined_Unused) {
  // If the combined var is in the interface, that is enough to trigger
  // its replacement. Otherwise the entry point interface is untouched
  // when the combined var is not otherwise used.
  const bool combined_var_in_interface =
      std::string(GetParam().initial_interface).find("%combined_var") !=
      std::string::npos;
  if (!combined_var_in_interface) {
    const std::string kTest = PreambleFragment(GetParam().initial_interface) +
                              R"(
  ; CHECK: OpEntryPoint Fragment %main "main")" +
                              GetParam().initial_interface  // Note this is the
                                                            // intial interface
                              + R"(
  ; These clauses ensure the expected interface is the whole interface.
  ; CHECK-NOT: %{{\d+}}
  ; CHECK-NOT: %in_var
  ; CHECK-NOT: %out_var
  ; CHECK-NOT: %combined_var
  ; CHECK: OpExecutionMode %main OriginUpperLeft

  ; All traces of the variable disappear
  ; CHECK-NOT: combined_var
  ; CHECK: OpFunctionEnd
                 OpName %combined_var "combined_var"
                 OpName %in_var "in_var"
                 OpName %out_var "out_var"
                 OpDecorate %combined_var DescriptorSet 0
                 OpDecorate %combined_var Binding 0
                 OpDecorate %in_var BuiltIn FragCoord
                 OpDecorate %out_var Location 0


                 %bool = OpTypeBool
  )" + BasicTypes() + R"(         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
           %11 = OpTypeSampledImage %10
  %_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11
       %in_ptr_v4f = OpTypePointer Input %v4float
       %in_var = OpVariable %in_ptr_v4f Input
      %out_ptr_v4f = OpTypePointer Output %v4float
      %out_var = OpVariable %out_ptr_v4f Output

  ; %combined_var is not used!
  %combined_var = OpVariable %_ptr_UniformConstant_11 UniformConstant
         %main = OpFunction %void None %voidfn
       %main_0 = OpLabel
                 OpReturn
                 OpFunctionEnd
)";
    auto [disasm, status] =
        SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
            kTest, /* do_validation= */ true);
    EXPECT_EQ(status, Pass::Status::SuccessWithChange) << disasm;
  }
}

INSTANTIATE_TEST_SUITE_P(EntryPointRemap,
                         SplitCombinedImageSamplerPassEntryPointRemapTest,
                         ::testing::ValuesIn(EntryPointInterfaceCases()));

// Remap function types

struct FunctionTypeCase {
  const char* initial_type_params = "";
  const char* expected_type_params = "";
};

std::ostream& operator<<(std::ostream& os, const FunctionTypeCase& ftc) {
  os << "(init " << ftc.initial_type_params << " -> expect "
     << ftc.expected_type_params << ")";
  return os;
}

struct SplitCombinedImageSamplerPassFunctionTypeTest
    : public PassTest<::testing::TestWithParam<FunctionTypeCase>> {
  virtual void SetUp() override {
    SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
                          SPV_BINARY_TO_TEXT_OPTION_INDENT |
                          SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  }
};

std::vector<FunctionTypeCase> FunctionTypeCases() {
  return std::vector<FunctionTypeCase>{
      {"", ""},
      {" %image_ty", " %image_ty"},
      {" %sampler_ty", " %sampler_ty"},
      {" %sampled_image_ty", " %image_ty %sampler_ty"},
      {" %uint %sampled_image_ty %float",
       " %uint %image_ty %sampler_ty %float"},
      {" %ptr_sampled_image_ty",
       " %_ptr_UniformConstant_image_ty %_ptr_UniformConstant_sampler_ty"},
      {" %uint %ptr_sampled_image_ty %float",
       " %uint %_ptr_UniformConstant_image_ty %_ptr_UniformConstant_sampler_ty "
       "%float"},
      {" %uint %ptr_sampled_image_ty %ptr_sampled_image_ty %float",
       " %uint %_ptr_UniformConstant_image_ty %_ptr_UniformConstant_sampler_ty "
       "%_ptr_UniformConstant_image_ty %_ptr_UniformConstant_sampler_ty "
       "%float"},
  };
};

TEST_P(SplitCombinedImageSamplerPassFunctionTypeTest, Samples) {
  const std::string kTest = Preamble() + +R"(
       OpName %sampler_ty "sampler_ty"
       OpName %image_ty "image_ty"
       OpName %f_ty "f_ty"
       OpName %sampled_image_ty "sampled_image_ty"
       OpName %ptr_sampled_image_ty "sampled_image_ty"

  )" + BasicTypes() + R"(

 %sampler_ty = OpTypeSampler
   %image_ty = OpTypeImage %float 2D 0 0 0 1 Unknown
 %sampled_image_ty = OpTypeSampledImage %image_ty
 %ptr_sampled_image_ty = OpTypePointer UniformConstant %sampled_image_ty

       %f_ty = OpTypeFunction %float)" +
                            GetParam().initial_type_params + R"(
       %bool = OpTypeBool

  ; CHECK: %f_ty = OpTypeFunction %float)" +
                            GetParam().expected_type_params + R"(
  ; CHECK-NEXT: %bool = OpTypeBool

         %main = OpFunction %void None %voidfn
       %main_0 = OpLabel
                 OpReturn
                 OpFunctionEnd
)";
  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest, /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange) << disasm;
}

INSTANTIATE_TEST_SUITE_P(FunctionTypeRemap,
                         SplitCombinedImageSamplerPassFunctionTypeTest,
                         ::testing::ValuesIn(FunctionTypeCases()));

// Remap function bodies

std::string NamedITypes(){
  return R"(
      OpName %f "f"
      OpName %f_ty "f_ty"
      OpName %i_ty "i_ty"
      OpName %s_ty "s_ty"
      OpName %si_ty "si_ty"
      OpName %p_i_ty "p_i_ty"
      OpName %p_s_ty "p_s_ty"
      OpName %p_si_ty "p_si_ty"
)";
}

std::string ITypes(){
  return R"(
      %i_ty = OpTypeImage %float 2D 0 0 0 1 Unknown
      %s_ty = OpTypeSampler
      %si_ty = OpTypeSampledImage %i_ty
      %p_i_ty = OpTypePointer UniformConstant %i_ty
      %p_s_ty = OpTypePointer UniformConstant %s_ty
      %p_si_ty = OpTypePointer UniformConstant %si_ty
)";
}

TEST_F(SplitCombinedImageSamplerPassTest, FunctionBody_ScalarNoChange) {
  const std::string kTest = Preamble() + NamedITypes() + BasicTypes() + ITypes() + R"(

      ; CHECK: %f_ty = OpTypeFunction %float %i_ty %s_ty %p_i_ty %p_s_ty
      %f_ty = OpTypeFunction %float %i_ty %s_ty %p_i_ty %p_s_ty

      ; CHECK: %f = OpFunction %float None %f_ty
      ; CHECK-NEXT: OpFunctionParameter %i_ty
      ; CHECK-NEXT: OpFunctionParameter %s_ty
      ; CHECK-NEXT: OpFunctionParameter %p_i_ty
      ; CHECK-NEXT: OpFunctionParameter %p_s_ty
      ; CHECK-NEXT: OpLabel
      %f = OpFunction %float None %f_ty
      %100 = OpFunctionParameter %i_ty
      %101 = OpFunctionParameter %s_ty
      %102 = OpFunctionParameter %p_i_ty
      %103 = OpFunctionParameter %p_s_ty
      %110 = OpLabel
      OpReturnValue %float_0
      OpFunctionEnd
      )" + Main();

  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest, /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange) << disasm;
}

TEST_F(SplitCombinedImageSamplerPassTest, FunctionBody_SampledImage) {
  const std::string kTest = Preamble() + NamedITypes() + BasicTypes() + ITypes() + R"(

      ; CHECK: %f_ty = OpTypeFunction %float %uint %i_ty %s_ty %float
      %f_ty = OpTypeFunction %float %uint %si_ty %float

      ; CHECK: %f = OpFunction %float None %f_ty
      ; CHECK-NEXT: OpFunctionParameter %uint
      ; CHECK-NEXT: %[[i:\w+]] = OpFunctionParameter %i_ty
      ; CHECK-NEXT: %[[s:\w+]] = OpFunctionParameter %s_ty
      ; CHECK-NEXT: OpFunctionParameter %float
      ; CHECK-NEXT: OpLabel
      ; CHECK-NEXT: %[[si:\w+]] = OpSampledImage %[[i]] %[[s]]
      ; CHECK-NEXT: %201 = %si_ty %[[si]]
      %f = OpFunction %float None %f_ty
      %100 = OpFunctionParameter %uint
      %101 = OpFunctionParameter %si_ty
      %110 = OpFunctionParameter %float
      %120 = OpLabel
      %201 = OpCopyObject %si_ty %101
      OpReturnValue %float_0
      OpFunctionEnd
      )" + Main();

  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest, /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange) << disasm;
}

TEST_F(SplitCombinedImageSamplerPassTest, DISABLED_FunctionBody_PtrSampledImage) {
  const std::string kTest = Preamble() + NamedITypes() + BasicTypes() + ITypes() + R"(

      ; CHECK: %f_ty = OpTypeFunction %float %uint %p_i_ty %p_s_ty %float
      %f_ty = OpTypeFunction %float %uint %p_si_ty %float

      ; CHECK: %f = OpFunction %float None %f_ty
      ; CHECK-NEXT: OpFunctionParameter %uint
      ; CHECK-NEXT: %[[pi:\w+]] = OpFunctionParameter %p_i_ty
      ; CHECK-NEXT: %[[ps:\w+]] = OpFunctionParameter %p_s_ty
      ; CHECK-NEXT: OpFunctionParameter %float
      ; CHECK-NEXT: OpLabel
      ; CHECK-NEXT: %[[i:\w+]] = OpLoad %i_ty %[[pi]]
      ; CHECK-NEXT: %[[s:\w+]] = OpLoad %s_ty %[[ps]]
      ; CHECK-NEXT: %[[si:\w+]] = OpSampledImage %[[i]] %[[s]]
      ; CHECK-NEXT: %130 = OpCopyObject %[[si]]
      %f = OpFunction %float None %f_ty
      %100 = OpFunctionParameter %uint
      %101 = OpFunctionParameter %p_si_ty
      %110 = OpFunctionParameter %float
      %120 = OpLabel
      %si = OpLoad %si_ty %101
      %130 = OpCopyObject %si_ty %si
      OpReturnValue %float_0
      OpFunctionEnd
      )" + Main();

  auto [disasm, status] = SinglePassRunAndMatch<SplitCombinedImageSamplerPass>(
      kTest, /* do_validation= */ true);
  EXPECT_EQ(status, Pass::Status::SuccessWithChange) << disasm;
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
