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
from . WordList import *

class TestWordList(unittest.TestCase):
  def test_creation_empty(self) -> None:
    x = WordList([])
    self.assertEqual(len(x), 0)
    self.assertEqual(x, [])

  def test_creation_nonempty(self) -> None:
    x = WordList([123, 456])
    self.assertEqual(len(x), 2)
    self.assertEqual(x, [123, 456])

  def test_creation_does_not_sort(self) -> None:
    x = WordList([123, 456])
    y = WordList([456, 123])
    self.assertEqual(len(x), 2)
    self.assertEqual(len(y), 2)
    self.assertEqual(x, [123, 456])
    self.assertEqual(y, [456, 123])

  def test_equality(self) -> None:
    x = WordList([123, 456])
    y = WordList([456, 123])
    self.assertEqual(x, x)
    self.assertNotEqual(x, y)

  def test_hash_heuristic(self) -> None:
    x = WordList([123, 456])
    y = WordList([123, 456])
    z = WordList([456, 123])
    self.assertEqual(hash(x), hash(y))
    self.assertNotEqual(hash(x), hash(z))

  def test_value_check(self) -> None:
    self.assertRaises(Exception, WordList, [-1])
    self.assertRaises(Exception, WordList, [1.5])
    self.assertRaises(Exception, WordList, [(1<<31)*2])
    WordList([(1<<32)-1])

if __name__ == "__main__":
    unittest.main()
