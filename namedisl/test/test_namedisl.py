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


def test_basic_set_from_str() -> None:
    bs = nisl.make_basic_set("[n] -> { [i]: 0 <= i < n }")
    print(bs._obj)
    print(bs)


def test_named_basic_set_from_basic_set() -> None:
    bset = isl.BasicSet("[n] -> { [i, j] : 0 <= i, j < n }")
    named_bset = nisl.make_basic_set(bset)
    print(named_bset._obj)
    print(named_bset)


def test_set_from_str() -> None:
    bs = nisl.make_set("[n] -> { [i]: 0 <= i < n }")
    print(bs._obj)
    print(bs)


def test_named_set_from_set() -> None:
    bset = isl.Set("[n] -> { [i, j] : 0 <= i, j < n }")
    named_bset = nisl.make_set(bset)
    print(named_bset._obj)
    print(named_bset)


def test_basic_map_from_str() -> None:
    bmap = nisl.make_basic_map(
        "[n] -> { [i0, j0] -> [i1, j1] : 0 <= i0, j0, i1, j1 < n }"
    )
    print(bmap._obj)
    print(bmap)


def test_named_basic_map_from_basic_map() -> None:
    bmap = isl.BasicMap(
        "[n] -> { [i0, j0] -> [i1, j1] : 0 <= i0, j0, i1, j1 < n }"
    )
    named_bmap = nisl.make_basic_map(bmap)
    print(named_bmap._obj)
    print(named_bmap)


def test_map_from_str() -> None:
    bmap = nisl.make_map(
        "[n] -> { [i0, j0] -> [i1, j1] : 0 <= i0, j0, i1, j1 < n }"
    )
    print(bmap._obj)
    print(bmap)


def test_named_map_from_map() -> None:
    bmap = isl.Map(
        "[n] -> { [i0, j0] -> [i1, j1] : 0 <= i0, j0, i1, j1 < n }"
    )
    named_bmap = nisl.make_map(bmap)
    print(named_bmap._obj)
    print(named_bmap)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
