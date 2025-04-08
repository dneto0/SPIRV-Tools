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

from typing import *
from . IndexRange import *

class Context():
    """
    Contains global tables for strings, and arrays of enums of various kinds.
    """
    def __init__(self) -> None:
        self.string_total_len: int = 0
        self.string_buffer: list[str] = []
        self.strings: dict[str, IndexRange] = {}

    def AddString(self, s: str) -> IndexRange:
        """
        Ensures string s is in the string buffer, adding it if absent.
        Returns its IndexRange.
        """
        if s in self.strings:
            return self.strings[s]
        # Allocate space, including for the terminating null.
        s_space: int = len(s) + 1
        ir = IndexRange(self.string_total_len, self.string_total_len + s_space)
        self.strings[s] = ir
        self.string_total_len += s_space
        self.string_buffer.append(s)
        return ir
