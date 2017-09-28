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

#include <gmock/gmock.h>

#include <memory>
#include <tuple>

#include "source/message.h"
#include "source/opt/make_unique.h"

#include "pass_fixture.h"
#include "pass_utils.h"

using spvtools::MakeUnique;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Not;

namespace {

using namespace spvtools;

struct GraphicsRobustAccessFailCase {
  // Assembly of input module.
  std::string input;
  // Expected error message substring.
  std::string message;
};

using GraphicsRobustAccessFailTest =
    spvtools::PassTest<::testing::TestWithParam<GraphicsRobustAccessFailCase>>;

TEST_P(GraphicsRobustAccessFailTest, FailTransform) {
  std::ostringstream errs;
  auto consumer = [&errs](spv_message_level_t l, const char* source,
                          const spv_position_t& pos, const char* msg) {
    errs << spvtools::StringifyMessage(l, source, pos, msg);
  };

  auto pass = MakeUnique<opt::GraphicsRobustAccessPass>();
  pass->SetMessageConsumer(consumer);

  auto result = OptimizeToBinary(pass.get(), GetParam().input, false);
  EXPECT_THAT(std::get<1>(result), Eq(spvtools::opt::Pass::Status::Failure));
  EXPECT_THAT(errs.str(), HasSubstr(GetParam().message));
}

INSTANTIATE_TEST_CASE_P(
    Sample, GraphicsRobustAccessFailTest,
    ::testing::ValuesIn(std::vector<GraphicsRobustAccessFailCase>{
        {"OpCapability VariablePointers",
         "Can't process module with VariablePointers capability"},
        {"OpCapability Shader\n"
         "OpMemoryModel Physical32 OpenCL\n",
         "Can't process module with addressing model other than Logical.  "
         "Found 1"},
        {"OpCapability Shader\n"
         "OpMemoryModel Physical64 OpenCL\n",
         "Can't process module with addressing model other than Logical.  "
         "Found 2"},

    }), );

struct GraphicsRobustAccessPassCase {
  // Assembly of input module.
  std::string input;
  // Expected output assembly.
  std::string output;
};

using GraphicsRobustAccessPassTest =
    spvtools::PassTest<::testing::TestWithParam<GraphicsRobustAccessPassCase>>;

TEST_P(GraphicsRobustAccessPassTest, PassTransform) {

  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                       SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);

  auto result = SinglePassRunAndDisassemble<opt::GraphicsRobustAccessPass>(
      GetParam().input, true);

  EXPECT_THAT(std::get<1>(result), Not(Eq(spvtools::opt::Pass::Status::Failure)));
  EXPECT_THAT(std::get<0>(result), Eq(GetParam().output));
}

INSTANTIATE_TEST_CASE_P(
    Sample, GraphicsRobustAccessPassTest,
    ::testing::ValuesIn(std::vector<GraphicsRobustAccessPassCase>{
        {"OpCapability Shader\nOpMemoryModel Logical GLSL450",
         "OpCapability Shader\nOpMemoryModel Logical GLSL450\n"},

    }), );



}  // anonymous namespace
