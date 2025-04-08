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
from enum import IntEnum
from . IndexRange import *
from . AliasList import *
from . WordList import *

class EnumKind(IntEnum):
    Capability = 1
    OperandType = 2
    Extension = 3

class Context():
    """
    Contains global tables for strings, and arrays of enums of various kinds.
    """
    def __init__(self) -> None:
        self.string_total_len: int = 0
        self.string_buffer: list[str] = []
        self.strings: dict[str, IndexRange] = {}

        # The concatenation of all alias lists.
        self.alias_buffer: list[IndexRange] = []
        # Maps an alias list to the subrange in self.aliasBuffer
        self.aliases: dict[AliasList, IndexRange] = {}

        self.enum_buffer: dict[EnumKind,list[int]] = {}
        self.enum_buffer[EnumKind.Capability] = []
        self.enum_buffer[EnumKind.OperandType] = []
        self.enum_buffer[EnumKind.Extension] = []
        self.enums: dict[EnumKind, dict[WordList, IndexRange]] = {}
        for k in EnumKind:
          self.enums[k] = {}

    def AddString(self, s: str) -> IndexRange:
        """
        Ensures string s is in the string table, adding it if absent.
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

    def AddAliasStringList(self, aliases: list[str]) -> IndexRange:
        """
        Ensures a list of strings exists in the alias table.
        Lists are first sorted before comparison or storage.
        A list is is represented as a list of IndexRange into the string table.
        Returns the IndexRange for the list itself.
        """
        l = AliasList([self.AddString(a) for a in sorted(aliases)])
        if l in self.aliases:
            return self.aliases[l]
        # Allocate space, including for the terminating null.
        ir = IndexRange(len(self.alias_buffer), len(l))
        self.alias_buffer.extend(l)
        self.aliases[l] = ir
        return ir

    def AddEnumList(self, kind: EnumKind, words: list[int]) -> IndexRange:
        """
        Ensures an ordered list of 32-bit integers (words) exists in the word table
        for the given enum kind.
        Returns the IndexRange for the list itself.
        """
        l = WordList(words)
        if l in self.enums[kind]:
             return self.enums[kind]
        ir = IndexRange(len(self.enum_buffer[kind]), len(l))
        self.enum_buffer[kind].extend(l)
        self.enums[kind][l] = ir
        return ir


