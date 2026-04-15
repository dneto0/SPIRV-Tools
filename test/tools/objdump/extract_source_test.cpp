// Copyright (c) 2023 Google LLC.
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

#include "tools/objdump/extract_source.h"

#include <gtest/gtest.h>

#include <string>

#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "spirv-tools/libspirv.hpp"
#include "test/test_fixture.h"
#include "tools/util/cli_consumer.h"

namespace {

constexpr auto kDefaultEnvironment = SPV_ENV_UNIVERSAL_1_6;

std::pair<bool, std::unordered_map<std::string, std::string>> ExtractSource(
    const std::string& spv_source, bool nonHostEndianness) {
  std::unique_ptr<spvtools::opt::IRContext> ctx = spvtools::BuildModule(
      kDefaultEnvironment, spvtools::utils::CLIMessageConsumer, spv_source,
      spvtools::SpirvTools::kDefaultAssembleOption |
          SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  std::vector<uint32_t> binary;
  ctx->module()->ToBinary(&binary, /* skip_nop = */ false);
  spvtest::MaybeFlipWords(nonHostEndianness, binary.begin(), binary.end());
  std::unordered_map<std::string, std::string> output;
  bool result = ExtractSourceFromModule(binary, &output);
  return std::make_pair(result, std::move(output));
}

using ExtractSourceTest = ::testing::TestWithParam<bool>;

TEST_P(ExtractSourceTest, no_debug) {
  std::string source = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
   %void = OpTypeVoid
      %2 = OpTypeFunction %void
   %bool = OpTypeBool
      %4 = OpUndef %bool
      %5 = OpFunction %void None %2
      %6 = OpLabel
           OpReturn
           OpFunctionEnd
  )";

  auto [success, result] = ExtractSource(source, GetParam());
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 0);
}

TEST_P(ExtractSourceTest, SimpleSource) {
  std::string source = R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
      OpEntryPoint GLCompute %1 "compute_1"
      OpExecutionMode %1 LocalSize 1 1 1
 %2 = OpString "compute.hlsl"
      OpSource HLSL 660 %2 "[numthreads(1, 1, 1)] void compute_1(){ }"
      OpName %1 "compute_1"
 %3 = OpTypeVoid
 %4 = OpTypeFunction %3
 %1 = OpFunction %3 None %4
 %5 = OpLabel
      OpLine %2 1 41
      OpReturn
      OpFunctionEnd
  )";

  auto [success, result] = ExtractSource(source, GetParam());
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  ASSERT_TRUE(result["compute.hlsl"] ==
              "[numthreads(1, 1, 1)] void compute_1(){ }");
}

TEST_P(ExtractSourceTest, SourceContinued) {
  std::string source = R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
      OpEntryPoint GLCompute %1 "compute_1"
      OpExecutionMode %1 LocalSize 1 1 1
 %2 = OpString "compute.hlsl"
      OpSource HLSL 660 %2 "[numthreads(1, 1, 1)] "
      OpSourceContinued "void compute_1(){ }"
      OpName %1 "compute_1"
 %3 = OpTypeVoid
 %4 = OpTypeFunction %3
 %1 = OpFunction %3 None %4
 %5 = OpLabel
      OpLine %2 1 41
      OpReturn
      OpFunctionEnd
  )";

  auto [success, result] = ExtractSource(source, GetParam());
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  ASSERT_TRUE(result["compute.hlsl"] ==
              "[numthreads(1, 1, 1)] void compute_1(){ }");
}

TEST_P(ExtractSourceTest, OnlyFilename) {
  std::string source = R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
      OpEntryPoint GLCompute %1 "compute_1"
      OpExecutionMode %1 LocalSize 1 1 1
 %2 = OpString "compute.hlsl"
      OpSource HLSL 660 %2
      OpName %1 "compute_1"
 %3 = OpTypeVoid
 %4 = OpTypeFunction %3
 %1 = OpFunction %3 None %4
 %5 = OpLabel
      OpLine %2 1 41
      OpReturn
      OpFunctionEnd
  )";

  auto [success, result] = ExtractSource(source, GetParam());
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  ASSERT_TRUE(result["compute.hlsl"] == "");
}

TEST_P(ExtractSourceTest, MultipleFiles) {
  std::string source = R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
      OpEntryPoint GLCompute %1 "compute_1"
      OpExecutionMode %1 LocalSize 1 1 1
 %2 = OpString "compute1.hlsl"
 %3 = OpString "compute2.hlsl"
      OpSource HLSL 660 %2 "some instruction"
      OpSource HLSL 660 %3 "some other instruction"
      OpName %1 "compute_1"
 %4 = OpTypeVoid
 %5 = OpTypeFunction %4
 %1 = OpFunction %4 None %5
 %6 = OpLabel
      OpLine %2 1 41
      OpReturn
      OpFunctionEnd
  )";

  auto [success, result] = ExtractSource(source, GetParam());
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 2);
  ASSERT_TRUE(result["compute1.hlsl"] == "some instruction");
  ASSERT_TRUE(result["compute2.hlsl"] == "some other instruction");
}

TEST_P(ExtractSourceTest, MultilineCode) {
  std::string source = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "compute_1"
               OpExecutionMode %1 LocalSize 1 1 1
          %2 = OpString "compute.hlsl"
               OpSource HLSL 660 %2 "[numthreads(1, 1, 1)]
void compute_1() {
}
"
               OpName %1 "compute_1"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %1 = OpFunction %3 None %4
          %5 = OpLabel
               OpLine %2 3 1
               OpReturn
               OpFunctionEnd
  )";

  auto [success, result] = ExtractSource(source, GetParam());
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  ASSERT_TRUE(result["compute.hlsl"] ==
              "[numthreads(1, 1, 1)]\nvoid compute_1() {\n}\n");
}

TEST_P(ExtractSourceTest, EmptyFilename) {
  std::string source = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "compute_1"
               OpExecutionMode %1 LocalSize 1 1 1
          %2 = OpString ""
               OpSource HLSL 660 %2 "void compute(){}"
               OpName %1 "compute_1"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %1 = OpFunction %3 None %4
          %5 = OpLabel
               OpLine %2 3 1
               OpReturn
               OpFunctionEnd
  )";

  auto [success, result] = ExtractSource(source, GetParam());
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  ASSERT_TRUE(result["unnamed-0.hlsl"] == "void compute(){}");
}

TEST_P(ExtractSourceTest, EscapeEscaped) {
  std::string source = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "compute"
               OpExecutionMode %1 LocalSize 1 1 1
          %2 = OpString "compute.hlsl"
               OpSource HLSL 660 %2 "// check \" escape removed"
               OpName %1 "compute"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %1 = OpFunction %3 None %4
          %5 = OpLabel
               OpLine %2 6 1
               OpReturn
               OpFunctionEnd
  )";

  auto [success, result] = ExtractSource(source, GetParam());
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  ASSERT_TRUE(result["compute.hlsl"] == "// check \" escape removed");
}

TEST_P(ExtractSourceTest, OpSourceWithNoSource) {
  std::string source = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "compute"
               OpExecutionMode %1 LocalSize 1 1 1
          %2 = OpString "compute.hlsl"
               OpSource HLSL 660 %2
               OpName %1 "compute"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %1 = OpFunction %3 None %4
          %5 = OpLabel
               OpLine %2 6 1
               OpReturn
               OpFunctionEnd
  )";

  auto [success, result] = ExtractSource(source, GetParam());
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  ASSERT_TRUE(result["compute.hlsl"] == "");
}

TEST_P(ExtractSourceTest, ExtendedInstructionSet) {
  std::string source = R"(
      OpCapability Shader
 %1 = OpExtInstImport "GLSL.std.450"
      OpMemoryModel Logical GLSL450
      OpEntryPoint GLCompute %11 "main"
      OpExecutionMode %1 LocalSize 1 1 1
 %2 = OpString "compute.hlsl"
      OpSource HLSL 660 %2 "[numthreads(1, 1, 1)] void main(){ }"
      OpName %1 "main"
 %3 = OpTypeVoid
 %4 = OpTypeFunction %3
 %9 = OpTypeFloat 32
%13 = OpConstant %9 3.14
%11 = OpFunction %3 None %4
 %5 = OpLabel
%10 = OpExtInst %9 %1 Sin %13
      OpReturn
      OpFunctionEnd
  )";

  auto [success, result] = ExtractSource(source, GetParam());
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  ASSERT_TRUE(result["compute.hlsl"] == "[numthreads(1, 1, 1)] void main(){ }")
      << result["compute.hlsl"];
}

TEST_P(ExtractSourceTest, Extension) {
  std::string source = R"(
      OpCapability Shader
      OpCapability Int8
      OpExtension "SPV_KHR_8bit_storage"
      OpMemoryModel Logical GLSL450
      OpEntryPoint GLCompute %main "main" %v
      OpExecutionMode %1 LocalSize 1 1 1
 %3 = OpTypeVoid
 %4 = OpTypeFunction %3
%char = OpTypeInt 8 1
%char_a = OpConstant %char 65
%pchar = OpTypePointer StorageBuffer %char
%v  = OpVariable %pchar StorageBuffer
%main = OpFunction %3 None %4
 %5 = OpLabel
      OpStore %v %char_a
      OpReturn
      OpFunctionEnd
  )";

  auto [success, result] = ExtractSource(source, GetParam());
  EXPECT_TRUE(success);
  ASSERT_EQ(result.size(), 0) << result.size();
}

INSTANTIATE_TEST_SUITE_P(HostEndian, ExtractSourceTest,
                         ::testing::Values(false));
INSTANTIATE_TEST_SUITE_P(OppositeEndian, ExtractSourceTest,
                         ::testing::Values(true));

}  // namespace
