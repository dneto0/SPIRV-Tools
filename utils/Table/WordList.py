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

import functools
from . IndexRange import IndexRange

class WordList(list):
  def __init__(self, words: list[int]):
    for w in words:
      if (w & 0xffffffff) != w:
        raise Exception("expected 32-bit unsigned word, got {}".format(w))
    super().__init__(words)

  def __hash__(self) -> int:
    return functools.reduce(lambda h, word: hash((h, word)), self, 0)
