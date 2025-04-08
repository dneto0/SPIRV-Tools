#!/usr/bin/env python3
# Copyright (c) 2025 Google Inc.

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
"""Manages a tables derived from the SPIR-V grammar."""

class Context():
    """
    Contains global tables for strings, and arrays of enums of various kinds.
    """
    def __init__(self):
        self.string_total_len = 0
        self.string_buffer = [] # ordered list of strings.
        self.strings = {} # Key is string, value is IndexRange

    def AddString(self, s):
        """
        Ensures string s is in the string buffer.
        """
        if s in self.strings:
            return self.strings[s]
        # Allocate space, including for the terminating null.
        s_space = len(s) + 1
        ir = IndexRange(self.string_total_len, self.string_total_len + s_space)
        self.strings[s] = ir
        self.string_total_len += s_space
        self.string_buffer.push(s)
        return ir
