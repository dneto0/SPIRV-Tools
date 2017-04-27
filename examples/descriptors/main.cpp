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

// This program demonstrates extraction of information about entry points
// in a module.

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "spirv-tools/libspirv.h"

#include "bin_to_descriptors_str.h"

// Prints a program usage message to stdout.
static void print_usage(const char* argv0) {
  printf(
      R"(%s - Show the descriptors used by entry points

Usage: %s [options] [<filename>]

The SPIR-V binary is read from <filename>. If no file is specified,
or if the filename is "-", then the binary is read from standard input.

Options:

  -h, --help      Print this help.
  --version       Display version information.
)",
      argv0, argv0);
}

int main(int argc, char** argv) {
  const char* inFile = nullptr;

  for (int argi = 1; argi < argc; ++argi) {
    if ('-' == argv[argi][0]) {
      switch (argv[argi][1]) {
        case 'h':
          print_usage(argv[0]);
          return 0;
        case '-': {
          // Long options
          if (0 == strcmp(argv[argi], "--help")) {
            print_usage(argv[0]);
            return 0;
          } else if (0 == strcmp(argv[argi], "--version")) {
            printf("%s EXPERIMENTAL\n", spvSoftwareVersionDetailsString());
            printf("Target: %s\n",
                   spvTargetEnvDescription(SPV_ENV_UNIVERSAL_1_1));
            return 0;
          } else {
            print_usage(argv[0]);
            return 1;
          }
        } break;
        case 0: {
          // Setting a filename of "-" to indicate stdin.
          if (!inFile) {
            inFile = argv[argi];
          } else {
            fprintf(stderr, "error: More than one input file specified\n");
            return 1;
          }
        } break;
        default:
          print_usage(argv[0]);
          return 1;
      }
    } else {
      if (!inFile) {
        inFile = argv[argi];
      } else {
        fprintf(stderr, "error: More than one input file specified\n");
        return 1;
      }
    }
  }

  // Read the input binary.
  std::stringstream input_stream;
  if (inFile) {
    std::ifstream input_file_stream(inFile,
                                    std::ios_base::in | std::ios_base::binary);
    if (input_file_stream.fail()) {
      std::cout << "error: Could not open " << inFile << " for reading"
                << std::endl;
      return 1;
    }
    input_file_stream >> input_stream.rdbuf();
  } else {
    std::cin >> input_stream.rdbuf();
  }

  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  spv_diagnostic diagnostic = nullptr;

  std::string contents(input_stream.str());
  std::ostringstream ss;
  auto error = BinaryToDescriptorsStr(
      context, reinterpret_cast<const uint32_t*>(contents.data()),
      contents.size() / 4, &ss, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    spvContextDestroy(context);
    return error;
  }
  std::cout << ss.str();

  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(context);

  return 0;
}
