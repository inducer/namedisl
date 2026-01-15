from __future__ import annotations


__copyright__ = """
Copyright (C) 2025- University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import islpy as isl

import namedisl as nisl


def test_set_from_str() -> None:
    spec = "[n] -> { [i] : 0 <= i < n }"
    s_isl = isl.Set(spec)
    s = nisl.make_set(spec)
    print(s)

    assert s._reconstruct_isl_object() == s_isl


def test_set_from_set() -> None:
    s_isl = isl.Set("[n] -> { [i] : 0 <= i < n }")
    s = nisl.make_set(s_isl)
    print(s)

    assert s._reconstruct_isl_object() == s_isl


def test_map_from_str() -> None:
    spec = "[n] -> { [i] -> [j] : 0 <= i < n and j = 2 * i }"
    m = nisl.make_map(spec)
    m_isl = isl.Map(spec)
    print(m)

    assert m._reconstruct_isl_object() == m_isl


def test_map_from_map() -> None:
    m_isl = isl.Map("[n] -> { [i] -> [j] : 0 <= i < n and j = 2 * i }")
    m = nisl.make_map(m_isl)
    print(m)

    assert m._reconstruct_isl_object() == m_isl
