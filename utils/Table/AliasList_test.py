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
from . Context import *
from . AliasList import *

class TestAliasList(unittest.TestCase):
  def test_creation_empty(self) -> None:
    x = AliasList([])
    self.assertEqual(len(x), 0)
    self.assertEqual(x, [])

  def test_creation_nonempty(self) -> None:
    c = Context()
    a = c.AddString("abc")
    b = c.AddString("def")
    x = AliasList([a, b])
    self.assertEqual(len(x), 2)
    self.assertEqual(x, [a, b])

  def test_creation_does_not_sort(self) -> None:
    c = Context()
    a = c.AddString("abc")
    b = c.AddString("def")
    x = AliasList([a, b])
    y = AliasList([b, a])
    self.assertNotEqual(x,y)

  def test_equality(self) -> None:
    c = Context()
    a = c.AddString("abc")
    b = c.AddString("def")
    x = AliasList([a, b])
    y = AliasList([a, b])
    z = AliasList([b, a])
    self.assertEqual(x, y)
    self.assertNotEqual(x, z)

  def test_hash_heuristic(self) -> None:
    c = Context()
    a = c.AddString("abc")
    b = c.AddString("def")
    x = AliasList([a, b])
    y = AliasList([a, b])
    z = AliasList([b, a])
    self.assertEqual(hash(x), hash(y))
    self.assertNotEqual(hash(x), hash(z))

if __name__ == "__main__":
    unittest.main()
