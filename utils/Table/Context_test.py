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
from Context import Context
from IndexRange import *

class TestContext(unittest.TestCase):
  def test_creation(self: object) -> None:
    x = Context()
    self.assertIsInstance(x.string_total_len, int)
    self.assertIsInstance(x.string_buffer, list)
    self.assertIsInstance(x.strings, dict)
    self.assertEqual(x.string_total_len, 0)
    self.assertEqual(x.string_buffer, [])
    self.assertEqual(x.strings, {})

  def test_AddString_new(self: object) -> None:
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

  def test_AddString_idempotent(self: object) -> None:
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


if __name__ == "__main__":
    unittest.main()
