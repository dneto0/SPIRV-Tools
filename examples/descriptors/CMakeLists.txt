# Copyright (c) 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_spvtools_example(
  TARGET spirv-tools-descriptors
  SRCS bin_to_descriptors_str.cpp
       entry_point_info.cpp
       main.cpp
  LIBS SPIRV-Tools
)
target_include_directories(spirv-tools-descriptors
  PRIVATE ${SPIRV-Headers_SOURCE_DIR}/include
)

add_spvtools_example(
  TARGET spirv-tools-entry-point-info-test
  SRCS entry_point_info.cpp
       entry_point_info_test.cpp
  LIBS SPIRV-Tools
       gmock
       gmock_main
)
if (NOT ${SPIRV_SKIP_EXECUTABLES})
  if (${COMPILER_IS_LIKE_GNU})
    target_compile_options(spirv-tools-entry-point-info-test PRIVATE -Wno-undef)
  endif()
  target_include_directories(spirv-tools-entry-point-info-test
        PRIVATE ${gmock_SOURCE_DIR}/include
                ${gtest_SOURCE_DIR}/include
                ${SPIRV-Headers_SOURCE_DIR}/include
                )
  add_test(NAME spirv-tools-entry-point-info-test
        COMMAND spirv-tools-entry-point-info-test)
endif()
