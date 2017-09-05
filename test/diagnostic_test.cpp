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

#include <gmock/gmock.h>

namespace {

using libspirv::DiagnosticStream;
using ::testing::Eq;

// Returns a newly created diagnostic value.
spv_diagnostic MakeValidDiagnostic() {
  spv_position_t position = {};
  spv_diagnostic diagnostic = spvDiagnosticCreate(&position, "");
  EXPECT_NE(nullptr, diagnostic);
  return diagnostic;
}

TEST(Diagnostic, DestroyNull) { spvDiagnosticDestroy(nullptr); }

TEST(Diagnostic, DestroyValidDiagnostic) {
  spv_diagnostic diagnostic = MakeValidDiagnostic();
  spvDiagnosticDestroy(diagnostic);
  // We aren't allowed to use the diagnostic pointer anymore.
  // So we can't test its behaviour.
}

TEST(Diagnostic, DestroyValidDiagnosticAfterReassignment) {
  spv_diagnostic diagnostic = MakeValidDiagnostic();
  spv_diagnostic second_diagnostic = MakeValidDiagnostic();
  EXPECT_TRUE(diagnostic != second_diagnostic);
  spvDiagnosticDestroy(diagnostic);
  diagnostic = second_diagnostic;
  spvDiagnosticDestroy(diagnostic);
}

TEST(Diagnostic, PrintDefault) {
  char message[] = "Test Diagnostic!";
  spv_diagnostic_t diagnostic = {{2, 3, 5}, message};
  // TODO: Redirect stderr
  ASSERT_EQ(SPV_SUCCESS, spvDiagnosticPrint(&diagnostic));
  // TODO: Validate the output of spvDiagnosticPrint()
  // TODO: Remove the redirection of stderr
}

TEST(Diagnostic, PrintInvalidDiagnostic) {
  ASSERT_EQ(SPV_ERROR_INVALID_DIAGNOSTIC, spvDiagnosticPrint(nullptr));
}

// TODO(dneto): We should be able to redirect the diagnostic printing.
// Once we do that, we can test diagnostic corner cases.

TEST(DiagnosticStream, ConversionToResultType) {
  // Check after the DiagnosticStream object is destroyed.
  spv_result_t value;
  { value = DiagnosticStream({}, nullptr, SPV_ERROR_INVALID_TEXT); }
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT, value);

  // Check implicit conversion via plain assignment.
  value = DiagnosticStream({}, nullptr, SPV_SUCCESS);
  EXPECT_EQ(SPV_SUCCESS, value);

  // Check conversion via constructor.
  EXPECT_EQ(SPV_FAILED_MATCH,
            spv_result_t(DiagnosticStream({}, nullptr, SPV_FAILED_MATCH)));
}

TEST(DiagnosticStream, EmitInfoToConsumer) {
  spvtest::MessageSink sink;
  {
    DiagnosticStream d({1, 2, 3}, sink.consumer(), SPV_SUCCESS);
    d << "hello world!";
  }

  EXPECT_THAT(sink.str(), Eq("4:input:{1 2 3}: hello world!\n"));
}

TEST(DiagnosticStream, EmitInfoAndNotesToConsumer) {
  // Notes accumulate at the end of the diagnostic stream.
  spvtest::MessageSink sink;
  {
    DiagnosticStream d({1, 2, 3}, sink.consumer(), SPV_SUCCESS);
    d << "hello world!" << libspirv::MakeNote("\nwith note: ")
      << libspirv::MakeNote(12) << " again";
  }

  EXPECT_THAT(sink.str(), Eq("4:input:{1 2 3}: hello world! again\nwith note: 12\n"));
}

TEST(DiagnosticStream, TextAndNotesSurviveMoving) {
  spvtest::MessageSink sink;
  {
    DiagnosticStream first({1, 2, 3}, sink.consumer(), SPV_SUCCESS);
    first << "hello world!" << libspirv::MakeNote("\nwith note: ")
          << libspirv::MakeNote(12) << " again";
    DiagnosticStream second(std::move(first));
    second << "(second)";
  }

  EXPECT_THAT(
      sink.str(),
      Eq("4:input:{1 2 3}: hello world! again(second)\nwith note: 12\n"));
}

}  // anonymous namespace
