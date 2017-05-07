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

#include <ostream>
#include <sstream>
#include <vector>

#include "spirv-tools/libspirv.hpp"

#include "entry_point_info.h"

namespace {

using spirv_example::Descriptor;
using spirv_example::Descriptors;
using spirv_example::EntryPointInfo;
using spirv_example::GetEntryPointInfo;
using testing::Eq;

using Infos = std::vector<EntryPointInfo>;
using Words = std::vector<uint32_t>;


// Returns a string representation of an EntryPointInfo object.
std::string Dump(const EntryPointInfo& e) {
  std::ostringstream out;
  out << "  EntryPoint(\"" << e.name() << "\":";
  for (const auto& d : e.descriptors()) {
    out << " (" << d.set << ", " << d.binding << ")";
  }
  out << std::endl;
  return out.str();
}

// Returns a string representation of a vector of EntryPointInfo objects to the
// given stream.
std::string Dump(const Infos& infos) {
  std::ostringstream out;
  out << "[" << std::endl;
  for (const auto& e : infos) {
    out << Dump(e) << "]" << std::endl;
  }
  return out.str();
}

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
  EXPECT_GE(result.size(), 5u) << source << "\n" << errs.str();
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

TEST(EntryPointInfo, ResetsThePassedInVector) {
  Infos infos(10); // Make non-empty.
  auto binary = Assemble(R"(
    OpCapability Addresses
    OpCapability Linkage
    OpCapability Kernel
    OpMemoryModel Physical32 OpenCL
  )");
  EXPECT_EQ(SPV_SUCCESS,
            GetEntryPointInfo(Context(), binary.data(), binary.size(), &infos,
                              nullptr));
  // Look. It was reset.
  EXPECT_THAT(infos, Eq(Infos{}));
}

TEST(EntryPointInfo, OneEntryPointTrivialBody) {
  Infos infos;
  auto binary = Assemble(R"(
    OpCapability Shader
    OpMemoryModel Logical GLSL450
    OpEntryPoint Vertex %main "foobar"
    %void = OpTypeVoid
    %void_fn = OpTypeFunction %void
    %main = OpFunction %void None %void_fn
    %entry = OpLabel
    OpReturn
    OpFunctionEnd
  )");
  EXPECT_EQ(SPV_SUCCESS,
            GetEntryPointInfo(Context(), binary.data(), binary.size(), &infos,
                              nullptr));
  EXPECT_THAT(infos, Eq(Infos{EntryPointInfo("foobar")}));
}

TEST(EntryPointInfo, SeveralEntryPointsTrivialBodies) {
  Infos infos;
  auto binary = Assemble(R"(
    OpCapability Shader
    OpMemoryModel Logical GLSL450
    OpEntryPoint Vertex %other " a first one! "
    OpEntryPoint Vertex %main "foobar"
    %void = OpTypeVoid
    %void_fn = OpTypeFunction %void
    %other = OpFunction %void None %void_fn
    %other_entry = OpLabel
    OpReturn
    OpFunctionEnd
    %main = OpFunction %void None %void_fn
    %entry = OpLabel
    OpReturn
    OpFunctionEnd
  )");
  EXPECT_EQ(SPV_SUCCESS, GetEntryPointInfo(Context(), binary.data(),
                                           binary.size(), &infos, nullptr));
  EXPECT_THAT(infos, Eq(Infos{EntryPointInfo(" a first one! "),
                              EntryPointInfo("foobar")}));
}

// Returns the preamble for a standard shader with a "main" Vertex shader.
std::string ShaderPreamble() {
  return R"(
    OpCapability Shader
    OpMemoryModel Logical GLSL450
    OpEntryPoint Vertex %main "main"
)";
}

std::string ShaderTypesAndConstants() {
  return R"(
    %void = OpTypeVoid
    %void_fn = OpTypeFunction %void
    %int = OpTypeInt 32 0
    %float = OpTypeFloat 32
    %zerof = OpConstantNull %float
    %zero = OpConstantNull %zero
    %one = OpConstant %int 1
    %float_unic_ptr = OpTypePointer UniformConstant %float
    %float_uni_ptr = OpTypePointer Uniform %float
    %float_sb_ptr = OpTypePointer StorageBuffer %float
    %float_arr = OpTypeRuntimeArray %float
    %float_arr_unic_ptr = OpTypePointer UniformConstant %float_arr
    %float_arr_uni_ptr = OpTypePointer Uniform %float_arr
    %float_arr_sb_ptr = OpTypePointer StorageBuffer %float_arr
    %struct_float = OpTypeStruct %float
    %struct_float_uni_ptr = OpTypePointer Uniform %struct_float
    %struct_float_sb_ptr = OpTypePointer StorageBuffer %struct_float
    %void_float_struct_float_sb_ptr_fn = OpTypeFunction %void %float %struct_float_sb_ptr
    %float_image_ptr = OpTypePointer Image %float
    %image_of_float = OpTypeImage %float 1D 0 0 0 2 R32f
    %image_of_float_unic_ptr = OpTypePointer UniformConstant %image_of_float
    %int_sb_ptr = OpTypePointer StorageBuffer %int
)";
}

TEST(EntryPointInfo, DirectlyReferencedViaLoad) {
  Infos infos;
  auto binary = Assemble(ShaderPreamble() +
                         R"(
    OpDecorate %var DescriptorSet 12
    OpDecorate %var Binding 8
)" + ShaderTypesAndConstants() + R"(
    %var = OpVariable %float_ptr UniformConstant
    %main = OpFunction %void None %void_fn
    %entry = OpLabel
    %value = OpLoad %float %var
    OpReturn
    OpFunctionEnd
)");
  EXPECT_EQ(SPV_SUCCESS, GetEntryPointInfo(Context(), binary.data(),
                                           binary.size(), &infos, nullptr));
  EXPECT_THAT(infos, Eq(Infos{EntryPointInfo("main", Descriptors{{12, 8}})}));
}

TEST(EntryPointInfo, DirectlyReferencedViaStore) {
  Infos infos;
  auto binary = Assemble(ShaderPreamble() +
                         R"(
    OpDecorate %var DescriptorSet 12
    OpDecorate %var Binding 18
)" + ShaderTypesAndConstants() + R"(
    %var = OpVariable %float_ptr UniformConstant
    %main = OpFunction %void None %void_fn
    %entry = OpLabel
    OpStore %var %zerof
    OpReturn
    OpFunctionEnd
)");
  EXPECT_EQ(SPV_SUCCESS, GetEntryPointInfo(Context(), binary.data(),
                                           binary.size(), &infos, nullptr));
  EXPECT_THAT(infos, Eq(Infos{EntryPointInfo("main", Descriptors{{12, 18}})}));
}

TEST(EntryPointInfo, DirectlyReferencedViaAccessChain) {
  Infos infos;
  auto binary = Assemble(ShaderPreamble() +
                         R"(
    OpDecorate %var DescriptorSet 9
    OpDecorate %var Binding 8
)" + ShaderTypesAndConstants() +
                         R"(
    %var = OpVariable %struct_float_uni_ptr Uniform
    %main = OpFunction %void None %void_fn
    %entry = OpLabel
    %p = OpAccessChain %float_uni_ptr %var %zero
    OpReturn
    OpFunctionEnd
)");
  EXPECT_EQ(SPV_SUCCESS, GetEntryPointInfo(Context(), binary.data(),
                                           binary.size(), &infos, nullptr));
  EXPECT_THAT(infos, Eq(Infos{EntryPointInfo("main", Descriptors{{9, 8}})}));
}

TEST(EntryPointInfo, DirectlyReferencedViaInBoundsAccessChain) {
  Infos infos;
  auto binary = Assemble(ShaderPreamble() +
                         R"(
    OpDecorate %var DescriptorSet 3
    OpDecorate %var Binding 2
)" + ShaderTypesAndConstants() +
                         R"(
    %var = OpVariable %struct_float_uni_ptr Uniform
    %main = OpFunction %void None %void_fn
    %entry = OpLabel
    %p = OpInBoundsAccessChain %float_uni_ptr %var %zero
    OpReturn
    OpFunctionEnd
)");
  EXPECT_EQ(SPV_SUCCESS, GetEntryPointInfo(Context(), binary.data(),
                                           binary.size(), &infos, nullptr));
  EXPECT_THAT(infos, Eq(Infos{EntryPointInfo("main", Descriptors{{3, 2}})}));
}

TEST(EntryPointInfo, DirectlyReferencedViaPtrAccessChain) {
  Infos infos;
  auto binary = Assemble(ShaderPreamble() +
                         R"(
    OpDecorate %var DescriptorSet 9
    OpDecorate %var Binding 8
)" + ShaderTypesAndConstants() +
                         R"(
    %var = OpVariable %struct_float_uni_ptr Uniform
    %main = OpFunction %void None %void_fn
    %entry = OpLabel
    %p = OpPtrAccessChain %float_uni_ptr %var %zero %zero
    OpReturn
    OpFunctionEnd
)");
  EXPECT_EQ(SPV_SUCCESS, GetEntryPointInfo(Context(), binary.data(),
                                           binary.size(), &infos, nullptr));
  EXPECT_THAT(infos, Eq(Infos{EntryPointInfo("main", Descriptors{{9, 8}})}));
}

TEST(EntryPointInfo, DirectlyReferencedViaInBoundsPtrAccessChain) {
  Infos infos;
  auto binary = Assemble(ShaderPreamble() +
                         R"(
    OpDecorate %var DescriptorSet 9
    OpDecorate %var Binding 8
)" + ShaderTypesAndConstants() +
                         R"(
    %var = OpVariable %struct_float_uni_ptr Uniform
    %main = OpFunction %void None %void_fn
    %entry = OpLabel
    %p = OpInBoundsPtrAccessChain %float_uni_ptr %var %zero %zero
    OpReturn
    OpFunctionEnd
)");
  EXPECT_EQ(SPV_SUCCESS, GetEntryPointInfo(Context(), binary.data(),
                                           binary.size(), &infos, nullptr));
  EXPECT_THAT(infos, Eq(Infos{EntryPointInfo("main", Descriptors{{9, 8}})}));
}

TEST(EntryPointInfo, DirectlyReferencedViaFunctionCall) {
  Infos infos;
  auto binary = Assemble(ShaderPreamble() +
                         R"(
    OpDecorate %var DescriptorSet 1
    OpDecorate %var Binding 3
)" + ShaderTypesAndConstants() +
                         R"(
    %var = OpVariable %struct_float_sb_ptr StorageBuffer

    %callee = OpFunction %void None %void_float_struct_float_sb_ptr_fn
    %callee_param0 = OpFunctionParameter %struct_float_sb_ptr_fn
    %callee_param1 = OpFunctionParameter %struct_float_sb_ptr_fn
    %callee_entry = OpLabel
    OpReturn

    %main = OpFunction %void None %void_fn
    %entry = OpLabel
    %res = OpFunctionCall %void %callee %zerof %var
    OpReturn
    OpFunctionEnd
)");
  EXPECT_EQ(SPV_SUCCESS, GetEntryPointInfo(Context(), binary.data(),
                                           binary.size(), &infos, nullptr));
  EXPECT_THAT(infos, Eq(Infos{EntryPointInfo("main", Descriptors{{1, 3}})}));
}

// A test case for finding descriptors via atomic operations.
struct DescriptorsCase {
  std::string set;
  std::string binding;
  std::string assembly;
  Descriptors expected;
};

using DescriptorsTest = ::testing::TestWithParam<DescriptorsCase>;

TEST_P(DescriptorsTest, Samples) {
  Infos infos;
  auto binary = Assemble(ShaderPreamble() + " OpDecorate %var DescriptorSet " +
                         GetParam().set + " OpDecorate %var Binding " +
                         GetParam().binding + " " + ShaderTypesAndConstants() +
                         R"(
    %var = OpVariable %int_sb_ptr StorageBuffer
    %var2 = OpVariable %int_sb_ptr StorageBuffer

    %main = OpFunction %void None %void_fn
    %entry = OpLabel
)" + GetParam().assembly +

                         R"(
    OpReturn
    OpFunctionEnd
)");
  EXPECT_EQ(SPV_SUCCESS, GetEntryPointInfo(Context(), binary.data(),
                                           binary.size(), &infos, nullptr))
      << Dump(infos);
  EXPECT_THAT(infos, Eq(Infos{EntryPointInfo("main", GetParam().expected)}));
}

INSTANTIATE_TEST_CASE_P(
    DirectlyReferencedViaAtomicOperation, DescriptorsTest,
    ::testing::ValuesIn(std::vector<DescriptorsCase>{
        {"3", "4", "%v = OpAtomicLoad %int %var %one %zero", {{3, 4}}},
        {"9", "7", "OpAtomicStore %var %one %zero %zero", {{9, 7}}},
        {"100",
         "92",
         "%v = OpAtomicExchange %int %var %one %zero %zero",
         {{100, 92}}},
        {"100",
         "92",
         "%v = OpAtomicCompareExchange %int %var %one %zero %zero %zero %zero",
         {{100, 92}}},
        {"10",
         "9",
         "%v = OpAtomicCompareExchangeWeak %int %var %one %zero %zero %zero "
         "%zero",
         {{10, 9}}},
        {"5", "6", "%v = OpAtomicIIncrement %int %var %one %zero", {{5, 6}}},
        {"6", "7", "%v = OpAtomicIDecrement %int %var %one %zero", {{6, 7}}},
        {"6", "9", "%v = OpAtomicIAdd %int %var %one %zero %one", {{6, 9}}},
        {"16", "19", "%v = OpAtomicISub %int %var %one %zero %one", {{16, 19}}},
        {"11", "12", "%v = OpAtomicSMin %int %var %one %zero %one", {{11, 12}}},
        {"13", "14", "%v = OpAtomicUMin %int %var %one %zero %one", {{13, 14}}},
        {"15", "16", "%v = OpAtomicSMax %int %var %one %zero %one", {{15, 16}}},
        {"17", "18", "%v = OpAtomicUMax %int %var %one %zero %one", {{17, 18}}},
        {"19", "20", "%v = OpAtomicAnd %int %var %one %zero %one", {{19, 20}}},
        {"21", "22", "%v = OpAtomicOr %int %var %one %zero %one", {{21, 22}}},
        {"23", "24", "%v = OpAtomicXor %int %var %one %zero %one", {{23, 24}}},
        {"25",
         "26",
         "%v = OpAtomicFlagTestAndSet %int %var %one %zero",
         {{25, 26}}},
        {"27", "28", "%v = OpAtomicFlagClear %var %one %zero", {{27, 28}}},
    }), );

INSTANTIATE_TEST_CASE_P(
    DirectlyReferencedViaCopy, DescriptorsTest,
    ::testing::ValuesIn(std::vector<DescriptorsCase>{
        {"99", "100", "%v = OpCopyObject %int_sb_ptr %var", {{99, 100}}},
        {"101", "102", "OpCopyMemory %var %var2", {{101, 102}}},
        {"103", "104", "OpCopyMemory %var2 %var", {{103, 104}}},
        {"103", "104", "OpCopyMemory %var2 %var", {{103, 104}}},
    }), );

TEST(EntryPointInfo, CaptureSeveralVariables) {
  Infos infos;
  auto binary = Assemble(ShaderPreamble() +
                         R"(
    OpDecorate %var DescriptorSet 12
    OpDecorate %var Binding 18
    OpDecorate %var2 DescriptorSet 13
    OpDecorate %var2 Binding 14
)" + ShaderTypesAndConstants() + R"(
    %var = OpVariable %float_ptr UniformConstant
    %var2 = OpVariable %float_ptr UniformConstant
    %main = OpFunction %void None %void_fn
    %entry = OpLabel
    OpCopyMemory %var %var2
    OpReturn
    OpFunctionEnd
)");
  EXPECT_EQ(SPV_SUCCESS, GetEntryPointInfo(Context(), binary.data(),
                                           binary.size(), &infos, nullptr));
  EXPECT_THAT(
      infos,
      Eq(Infos{EntryPointInfo("main", Descriptors{{12, 18}, {13, 14}})}));
}


}  // anonymous namespace
