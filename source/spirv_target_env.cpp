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

#include "source/spirv_target_env.h"

#include <cassert>
#include <cctype>
#include <cstring>
#include <string>

#include "source/spirv_constant.h"
#include "spirv-tools/libspirv.h"

const char* spvTargetEnvDescription(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
      return "SPIR-V 1.0";
    case SPV_ENV_VULKAN_1_0:
      return "SPIR-V 1.0 (under Vulkan 1.0 semantics)";
    case SPV_ENV_UNIVERSAL_1_1:
      return "SPIR-V 1.1";
    case SPV_ENV_OPENCL_1_2:
      return "SPIR-V 1.0 (under OpenCL 1.2 Full Profile semantics)";
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
      return "SPIR-V 1.0 (under OpenCL 1.2 Embedded Profile semantics)";
    case SPV_ENV_OPENCL_2_0:
      return "SPIR-V 1.0 (under OpenCL 2.0 Full Profile semantics)";
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
      return "SPIR-V 1.0 (under OpenCL 2.0 Embedded Profile semantics)";
    case SPV_ENV_OPENCL_2_1:
      return "SPIR-V 1.0 (under OpenCL 2.1 Full Profile semantics)";
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
      return "SPIR-V 1.0 (under OpenCL 2.1 Embedded Profile semantics)";
    case SPV_ENV_OPENCL_2_2:
      return "SPIR-V 1.2 (under OpenCL 2.2 Full Profile semantics)";
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
      return "SPIR-V 1.2 (under OpenCL 2.2 Embedded Profile semantics)";
    case SPV_ENV_OPENGL_4_0:
      return "SPIR-V 1.0 (under OpenGL 4.0 semantics)";
    case SPV_ENV_OPENGL_4_1:
      return "SPIR-V 1.0 (under OpenGL 4.1 semantics)";
    case SPV_ENV_OPENGL_4_2:
      return "SPIR-V 1.0 (under OpenGL 4.2 semantics)";
    case SPV_ENV_OPENGL_4_3:
      return "SPIR-V 1.0 (under OpenGL 4.3 semantics)";
    case SPV_ENV_OPENGL_4_5:
      return "SPIR-V 1.0 (under OpenGL 4.5 semantics)";
    case SPV_ENV_UNIVERSAL_1_2:
      return "SPIR-V 1.2";
    case SPV_ENV_UNIVERSAL_1_3:
      return "SPIR-V 1.3";
    case SPV_ENV_VULKAN_1_1:
      return "SPIR-V 1.3 (under Vulkan 1.1 semantics)";
    case SPV_ENV_WEBGPU_0:
      assert(false && "Deprecated target environment value.");
      break;
    case SPV_ENV_UNIVERSAL_1_4:
      return "SPIR-V 1.4";
    case SPV_ENV_VULKAN_1_1_SPIRV_1_4:
      return "SPIR-V 1.4 (under Vulkan 1.1 semantics)";
    case SPV_ENV_UNIVERSAL_1_5:
      return "SPIR-V 1.5";
    case SPV_ENV_VULKAN_1_2:
      return "SPIR-V 1.5 (under Vulkan 1.2 semantics)";
    case SPV_ENV_UNIVERSAL_1_6:
      return "SPIR-V 1.6";
    case SPV_ENV_VULKAN_1_3:
      return "SPIR-V 1.6 (under Vulkan 1.3 semantics)";
    case SPV_ENV_MAX:
      assert(false && "Invalid target environment value.");
      break;
  }
  return "";
}

uint32_t spvVersionForTargetEnv(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
      return SPV_SPIRV_VERSION_WORD(1, 0);
    case SPV_ENV_UNIVERSAL_1_1:
      return SPV_SPIRV_VERSION_WORD(1, 1);
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_OPENCL_2_2:
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
      return SPV_SPIRV_VERSION_WORD(1, 2);
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_VULKAN_1_1:
      return SPV_SPIRV_VERSION_WORD(1, 3);
    case SPV_ENV_WEBGPU_0:
      assert(false && "Deprecated target environment value.");
      break;
    case SPV_ENV_UNIVERSAL_1_4:
    case SPV_ENV_VULKAN_1_1_SPIRV_1_4:
      return SPV_SPIRV_VERSION_WORD(1, 4);
    case SPV_ENV_UNIVERSAL_1_5:
    case SPV_ENV_VULKAN_1_2:
      return SPV_SPIRV_VERSION_WORD(1, 5);
    case SPV_ENV_UNIVERSAL_1_6:
    case SPV_ENV_VULKAN_1_3:
      return SPV_SPIRV_VERSION_WORD(1, 6);
    case SPV_ENV_MAX:
      assert(false && "Invalid target environment value.");
      break;
  }
  return SPV_SPIRV_VERSION_WORD(0, 0);
}

bool spvParseTargetEnv(const char* s, spv_target_env* env) {
  auto match = [s](const char* b) {
    return s && (0 == strncmp(s, b, strlen(b)));
  };
  for (auto& name_env : spvTargetEnvNameMap) {
    if (match(name_env.first)) {
      if (env) {
        *env = name_env.second;
      }
      return true;
    }
  }
  if (env) *env = SPV_ENV_UNIVERSAL_1_0;
  return false;
}

bool spvReadEnvironmentFromText(std::vector<char>& text, spv_target_env* env) {
  // Version is expected to match "; Version: 1.X"
  // Version string must occur in header, that is, initial lines of comments
  // Once a non-comment line occurs, the header has ended
  for (std::size_t i = 0; i < text.size(); ++i) {
    char c = text[i];

    if (c == ';') {
      // Try to match against the expected version string
      constexpr const char* kVersionPrefix = "; Version: 1.";
      constexpr const auto kPrefixLength = 13;
      constexpr const auto kMinimumVersionLength = kPrefixLength + 2;
      if (i + kMinimumVersionLength >= text.size()) return false;

      auto j = 1;
      for (; j < kPrefixLength; ++j) {
        if (kVersionPrefix[j] != text[i + j]) break;
      }
      // j will match the prefix length if all characters before matched
      if (j == kPrefixLength) {
        char minor = text[i + j];
        if (minor >= '0' && minor <= '6' && !std::isdigit(text[i + j + 1])) {
          auto offset = minor - '0';
          *env = spvTargetEnvNameMap[kSpvEnvUniversalStart + offset].second;
          return true;
        }
      }

      // If no match, determine whether the header has ended (in which case,
      // assumption has failed.)
      i = j;
      for (; i < text.size(); ++i) {
        if (text[i] == '\n') break;
      }
    } else if (!std::isspace(c))
      break;
  }
  return false;
}

#define VULKAN_VER(MAJOR, MINOR) ((MAJOR << 22) | (MINOR << 12))
#define SPIRV_VER(MAJOR, MINOR) ((MAJOR << 16) | (MINOR << 8))

struct VulkanEnv {
  spv_target_env vulkan_env;
  uint32_t vulkan_ver;
  uint32_t spirv_ver;
};
// Maps each Vulkan target environment enum to the Vulkan version, and the
// maximum supported SPIR-V version for that Vulkan environment.
// Keep this ordered from least capable to most capable.
static const VulkanEnv ordered_vulkan_envs[] = {
    {SPV_ENV_VULKAN_1_0, VULKAN_VER(1, 0), SPIRV_VER(1, 0)},
    {SPV_ENV_VULKAN_1_1, VULKAN_VER(1, 1), SPIRV_VER(1, 3)},
    {SPV_ENV_VULKAN_1_1_SPIRV_1_4, VULKAN_VER(1, 1), SPIRV_VER(1, 4)},
    {SPV_ENV_VULKAN_1_2, VULKAN_VER(1, 2), SPIRV_VER(1, 5)},
    {SPV_ENV_VULKAN_1_3, VULKAN_VER(1, 3), SPIRV_VER(1, 6)}};

bool spvParseVulkanEnv(uint32_t vulkan_ver, uint32_t spirv_ver,
                       spv_target_env* env) {
  for (auto triple : ordered_vulkan_envs) {
    if (triple.vulkan_ver >= vulkan_ver && triple.spirv_ver >= spirv_ver) {
      *env = triple.vulkan_env;
      return true;
    }
  }
  return false;
}

bool spvIsVulkanEnv(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_OPENCL_2_2:
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_UNIVERSAL_1_4:
    case SPV_ENV_UNIVERSAL_1_5:
    case SPV_ENV_UNIVERSAL_1_6:
      return false;
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_VULKAN_1_1:
    case SPV_ENV_VULKAN_1_1_SPIRV_1_4:
    case SPV_ENV_VULKAN_1_2:
    case SPV_ENV_VULKAN_1_3:
      return true;
    case SPV_ENV_WEBGPU_0:
      assert(false && "Deprecated target environment value.");
      break;
    case SPV_ENV_MAX:
      assert(false && "Invalid target environment value.");
      break;
  }
  return false;
}

bool spvIsOpenCLEnv(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_VULKAN_1_1:
    case SPV_ENV_UNIVERSAL_1_4:
    case SPV_ENV_VULKAN_1_1_SPIRV_1_4:
    case SPV_ENV_UNIVERSAL_1_5:
    case SPV_ENV_VULKAN_1_2:
    case SPV_ENV_UNIVERSAL_1_6:
    case SPV_ENV_VULKAN_1_3:
      return false;
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_2_2:
      return true;
    case SPV_ENV_WEBGPU_0:
      assert(false && "Deprecated target environment value.");
      break;
    case SPV_ENV_MAX:
      assert(false && "Invalid target environment value.");
      break;
  }
  return false;
}

bool spvIsOpenGLEnv(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_VULKAN_1_1:
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_2_2:
    case SPV_ENV_UNIVERSAL_1_4:
    case SPV_ENV_VULKAN_1_1_SPIRV_1_4:
    case SPV_ENV_UNIVERSAL_1_5:
    case SPV_ENV_VULKAN_1_2:
    case SPV_ENV_UNIVERSAL_1_6:
    case SPV_ENV_VULKAN_1_3:
      return false;
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
      return true;
    case SPV_ENV_WEBGPU_0:
      assert(false && "Deprecated target environment value.");
      break;
    case SPV_ENV_MAX:
      assert(false && "Invalid target environment value.");
      break;
  }
  return false;
}

bool spvIsValidEnv(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_VULKAN_1_1:
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_2:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_2_2:
    case SPV_ENV_UNIVERSAL_1_4:
    case SPV_ENV_VULKAN_1_1_SPIRV_1_4:
    case SPV_ENV_UNIVERSAL_1_5:
    case SPV_ENV_VULKAN_1_2:
    case SPV_ENV_UNIVERSAL_1_6:
    case SPV_ENV_VULKAN_1_3:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
      return true;
    case SPV_ENV_WEBGPU_0:
    case SPV_ENV_MAX:
      break;
  }
  return false;
}

std::string spvLogStringForEnv(spv_target_env env) {
  switch (env) {
    case SPV_ENV_OPENCL_1_2:
    case SPV_ENV_OPENCL_2_0:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_2_2:
    case SPV_ENV_OPENCL_EMBEDDED_1_2:
    case SPV_ENV_OPENCL_EMBEDDED_2_0:
    case SPV_ENV_OPENCL_EMBEDDED_2_1:
    case SPV_ENV_OPENCL_EMBEDDED_2_2: {
      return "OpenCL";
    }
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5: {
      return "OpenGL";
    }
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_VULKAN_1_1:
    case SPV_ENV_VULKAN_1_1_SPIRV_1_4:
    case SPV_ENV_VULKAN_1_2:
    case SPV_ENV_VULKAN_1_3: {
      return "Vulkan";
    }
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_UNIVERSAL_1_4:
    case SPV_ENV_UNIVERSAL_1_5:
    case SPV_ENV_UNIVERSAL_1_6: {
      return "Universal";
    }
    case SPV_ENV_WEBGPU_0:
      assert(false && "Deprecated target environment value.");
      break;
    case SPV_ENV_MAX:
      assert(false && "Invalid target environment value.");
      break;
  }
  return "Unknown";
}

std::string spvTargetEnvList(const int pad, const int wrap) {
  std::string ret;
  size_t max_line_len = wrap - pad;  // The first line isn't padded
  std::string line;
  std::string sep = "";

  for (auto& name_env : spvTargetEnvNameMap) {
    std::string word = sep + name_env.first;
    if (line.length() + word.length() > max_line_len) {
      // Adding one word wouldn't fit, commit the line in progress and
      // start a new one.
      ret += line + "\n";
      line.assign(pad, ' ');
      // The first line is done. The max length now comprises the
      // padding.
      max_line_len = wrap;
    }
    line += word;
    sep = "|";
  }

  ret += line;

  return ret;
}
