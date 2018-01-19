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

#include "unit_spirv.h"

#include "gmock/gmock.h"
#include "test_fixture.h"

namespace {

using spvtest::MakeVector;
using spvtest::ScopedContext;
using ::testing::Eq;
using Words = std::vector<uint32_t>;

TEST(MakeVector, Samples) {
  EXPECT_THAT(MakeVector(""), Eq(Words{0}));
  EXPECT_THAT(MakeVector("a"), Eq(Words{0x0061}));
  EXPECT_THAT(MakeVector("ab"), Eq(Words{0x006261}));
  EXPECT_THAT(MakeVector("abc"), Eq(Words{0x00636261}));
  EXPECT_THAT(MakeVector("abcd"), Eq(Words{0x64636261, 0x00}));
  EXPECT_THAT(MakeVector("abcde"), Eq(Words{0x64636261, 0x0065}));
}

TEST(WordVectorPrintTo, PreservesFlagsAndFill) {
  std::stringstream s;
  s << std::setw(4) << std::oct << std::setfill('x') << 8 << " ";
  spvtest::PrintTo(spvtest::WordVector({10, 16}), &s);
  // The octal setting and fill character should be preserved
  // from before the PrintTo.
  // Width is reset after each emission of a regular scalar type.
  // So set it explicitly again.
  s << std::setw(4) << 9;

  EXPECT_THAT(s.str(), Eq("xx10 0x0000000a 0x00000010 xx11"));
}

TEST_P(RoundTripTest, Sample) {
  EXPECT_THAT(EncodeAndDecodeSuccessfully(GetParam()), Eq(GetParam()))
      << GetParam();
}

// spvtest::ScopedContext

TEST(ScopedContext, DefaultsToUniversal1_0) {
  ScopedContext ctx;
  ASSERT_NE(nullptr, ctx.context);
  EXPECT_EQ(SPV_ENV_UNIVERSAL_1_0, ctx.context->target_env);
}

TEST(ScopedContext, CanBeSetToUniversal1_2) {
  ScopedContext ctx(SPV_ENV_UNIVERSAL_1_2);
  ASSERT_NE(nullptr, ctx.context);
  EXPECT_EQ(SPV_ENV_UNIVERSAL_1_2, ctx.context->target_env);
}

TEST(ScopedContext, MoveConstuctorSetsOtherToNull) {
  ScopedContext ctx(SPV_ENV_UNIVERSAL_1_1);
  ScopedContext ctx2(std::move(ctx));
  EXPECT_EQ(nullptr, ctx.context);
  ASSERT_NE(nullptr, ctx2.context);
  EXPECT_EQ(SPV_ENV_UNIVERSAL_1_1, ctx2.context->target_env);
}

TEST(ScopedContext, MoveAssignmentSetsOtherToNull) {
  ScopedContext ctx(SPV_ENV_VULKAN_1_0);
  ScopedContext ctx2 = std::move(ctx);
  EXPECT_EQ(nullptr, ctx.context);
  ASSERT_NE(nullptr, ctx2.context);
  EXPECT_EQ(SPV_ENV_VULKAN_1_0, ctx2.context->target_env);
}

}  // anonymous namespace
