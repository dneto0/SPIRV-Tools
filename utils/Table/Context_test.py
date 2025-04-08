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

import unittest
from . Context import Context, EnumKind
from . IndexRange import IndexRange
from . AliasList import AliasList
from . WordList import WordList

class TestCreate(unittest.TestCase):
  def test_creation(self) -> None:
    x = Context()
    self.assertIsInstance(x.string_total_len, int)
    self.assertIsInstance(x.string_buffer, list)
    self.assertIsInstance(x.strings, dict)
    self.assertEqual(x.string_total_len, 0)
    self.assertEqual(x.string_buffer, [])
    self.assertEqual(x.strings, {})

class TestString(unittest.TestCase):
  def test_AddString_new(self) -> None:
    x = Context()
    abc_ir = x.AddString("abc")
    self.assertEqual(abc_ir, IndexRange(0,4))
    self.assertEqual(x.string_total_len, 4)
    self.assertEqual(x.string_buffer, ["abc"])
    self.assertEqual(x.strings, {"abc": IndexRange(0,4)})

    qz_ir = x.AddString("qz")
    self.assertEqual(qz_ir, IndexRange(4,7))
    self.assertEqual(x.string_total_len, 7)
    self.assertEqual(x.string_buffer, ["abc", "qz"])
    self.assertEqual(x.strings, {"abc": IndexRange(0,4), "qz": IndexRange(4,7)})

    empty_ir = x.AddString("")
    self.assertEqual(empty_ir, IndexRange(7,8))
    self.assertEqual(x.string_total_len, 8)
    self.assertEqual(x.string_buffer, ["abc", "qz", ""])
    self.assertEqual(x.strings, {"abc": IndexRange(0,4), "qz": IndexRange(4,7), "": IndexRange(7,8)})

  def test_AddString_idempotent(self) -> None:
    x = Context()
    abc_ir = x.AddString("abc")
    self.assertEqual(abc_ir, IndexRange(0,4))
    self.assertEqual(x.string_total_len, 4)
    self.assertEqual(x.string_buffer, ["abc"])
    self.assertEqual(x.strings, {"abc": IndexRange(0,4)})

    abc_ir = x.AddString("abc")
    self.assertEqual(abc_ir, IndexRange(0,4))
    self.assertEqual(x.string_total_len, 4)
    self.assertEqual(x.string_buffer, ["abc"])
    self.assertEqual(x.strings, {"abc": IndexRange(0,4)})

class TestAliases(unittest.TestCase):
  def test_AddAliasStringList_empty(self) -> None:
    x = Context()
    x_ir = x.AddAliasStringList([])
    self.assertEqual(x_ir, IndexRange(0,0))
    self.assertEqual(x.string_buffer, [])
    self.assertEqual(x.alias_buffer, [])
    self.assertEqual(x.alias_buffer, [])
    self.assertEqual(x.aliases, {AliasList([]): IndexRange(0,0)})

  def test_AddAliasStringList_nonempty_sorts(self) -> None:
    x = Context()
    a_ir = x.AddString("abc")
    b_ir = x.AddString("def")
    x_ir = x.AddAliasStringList(["def", "abc"])
    y_ir = x.AddAliasStringList(["abc", "def"])

    self.assertEqual(x_ir, IndexRange(0,2))
    self.assertEqual(y_ir, IndexRange(0,2))
    self.assertEqual(x.alias_buffer, [a_ir, b_ir])
    al = AliasList([a_ir, b_ir])
    self.assertEqual(x.aliases, {al: IndexRange(0,2)})

  def test_AddAliasStringList_nonempty_twice(self) -> None:
    x = Context()
    a_ir = x.AddString("abc")
    b_ir = x.AddString("def")
    c_ir = x.AddString("ghi")
    x_ir = x.AddAliasStringList(["abc", "def"])
    y_ir = x.AddAliasStringList(["abc", "ghi"])

    self.assertEqual(x_ir, IndexRange(0,2))
    self.assertEqual(y_ir, IndexRange(2,2))
    self.assertEqual(x.alias_buffer, [a_ir, b_ir, a_ir, c_ir])
    abl = AliasList([a_ir, b_ir])
    acl = AliasList([a_ir, c_ir])
    self.assertEqual(x.aliases, {abl: IndexRange(0,2), acl: IndexRange(2,2)})

class TestEnums(unittest.TestCase):
  def test_AddEnumList_empty(self) -> None:
    for k in EnumKind:
        c = Context()
        x_ir = c.AddEnumList(k, [])
        self.assertEqual(x_ir, IndexRange(0,0))
        self.assertEqual(c.enum_buffer[k], [])
        self.assertEqual(c.enums[k], {WordList([]): IndexRange(0,0)})

  def test_AddEnumList_nonempty_does_not_sort(self) -> None:
    for k in EnumKind:
        c = Context()
        x_ir = c.AddEnumList(k, [123,456])
        y_ir = c.AddEnumList(k, [456,123])
        self.assertEqual(x_ir, IndexRange(0,2))
        self.assertEqual(y_ir, IndexRange(2,2))
        self.assertEqual(c.enum_buffer[k], [123, 456, 456, 123])
        self.assertEqual(c.enums[k], { WordList([123,456]): x_ir, WordList([456, 123]): y_ir})

    def test_AddAliasStringList_idempotent(self) -> None:
      for k in EnumKind:
          c = Context()
          x_ir = c.AddEnumList(k, [123,456])
          y_ir = c.AddEnumList(k, [123,456])
          self.assertEqual(x_ir, IndexRange(0,2))
          self.assertEqual(y_ir, IndexRange(0,2))
          self.assertEqual(c.enum_buffer[k], [123, 456])
          self.assertEqual(c.enums, { WordList([123,456]): x_ir })

    def test_AddEnumList_enum_kinds_disjoint(self) -> None:
      c = Context()
      i: int = 0
      for k in EnumKind:
          i += 1
          x_ir = c.AddEnumList(k, [123,456 + i])
          self.assertEqual(x_ir, IndexRange(0,2))
          self.assertEqual(c.enum_buffer[k], [123, 456 + i])
          self.assertEqual(c.enums, { WordList([123,456 + i]): x_ir})
      self.assertEqual(c.enum_buffer[EnumKind.Capability], [123, 457])
      self.assertEqual(c.enum_buffer[EnumKind.OperandType], [123, 458])
      self.assertEqual(c.enum_buffer[EnumKind.Extension], [123, 459])

    def test_AddEnumList_value_check(self) -> None:
      for k in EnumKind:
          c = Context()
          c.AddEnumList(k, [(1<<32)-1])
          self.assertRaises(Exception, c.AddEnumList([-1]))
          self.assertRaises(Exception, c.AddEnumList([1.5]))
          self.assertRaises(Exception, c.AddEnumList([1<<32]))

if __name__ == "__main__":
    unittest.main()
