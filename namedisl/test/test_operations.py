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

import pytest

import namedisl as nisl


@pytest.mark.parametrize("ndims", [1, 2, 8])
def test_basic_set_intersection(ndims) -> None:
    dims = ",".join(f"i{d}" for d in range(ndims))
    a = nisl.make_basic_set(f"[n] -> {{ [{dims}] : 0 <= {dims} < n }}")
    b = nisl.make_basic_set(f"[n] -> {{ [{dims}] : 0 <= {dims} < 2*n }}")

    assert (a & b)._obj == (a._obj & b._obj)


@pytest.mark.parametrize("ndims", [1, 2, 8])
def test_basic_map_intersection(ndims) -> None:
    in_dims = ",".join(f"i{d}" for d in range(ndims))
    out_dims = ",".join(f"j{d}" for d in range(ndims))

    a = nisl.make_basic_map(
        f"[n] -> {{ [{in_dims}] -> [{out_dims}]: 0 <= {in_dims}, {out_dims} < n}}"
    )
    b = nisl.make_basic_map(
        f"[n] -> {{ [{in_dims}] -> [{out_dims}]: 0 <= {in_dims}, {out_dims} < n}}"
    )

    assert (a & b)._obj == (a._obj & b._obj)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[0])
    else:
        from pytest import main
        main([__file__])
