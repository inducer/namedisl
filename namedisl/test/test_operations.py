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


def _generate_dims_and_random_conditions(n, has_params):
    from random import randint 

    a_dims = [f"i{i}" for i in range(n)]
    b_dims = [f"j{i}" for i in range(n)]

    if not has_params:
        a_cond = " and ".join(f"0 <= {a} <= {randint(0, 100)}" for a in a_dims)
        b_cond = " and ".join(f"0 <= {b} <= {randint(0, 100)}" for b in b_dims)
    else:
        a_cond = ""
        b_cond = ""

    a_dim_string = ",".join(a for a in a_dims)
    b_dim_string = ",".join(b for b in b_dims)

    return a_dim_string, a_cond, b_dim_string, b_cond


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
@pytest.mark.parametrize("has_params", [True, False])
def test_set_union(ndims, has_params) -> None:
    a_dims, a_cond, b_dims, b_cond = _generate_dims_and_random_conditions(
        ndims, has_params)

    if has_params:
        a = nisl.make_set(f"[n] -> {{ [{a_dims}] : 0 <= {a_dims} < n }}")
        b = nisl.make_set(f"[m] -> {{ [{b_dims}] : 0 <= {b_dims} < m }}")
        result = nisl.make_set(
            f"""
            [n, m] -> {{ [{a_dims}, {b_dims}] : (0 <= {a_dims} < n) or (0 <= {b_dims} < m)}}
            """
        )
    else:
        a = nisl.make_set(f"{{ [{a_dims}] : {a_cond} }}")
        b = nisl.make_set(f"{{ [{b_dims}] : {b_cond} }}")
        result = nisl.make_set(
            f"""
            {{ [{a_dims}, {b_dims}] : ({a_cond}) or ({b_cond})}}
            """
        )

    union = a | b
    assert union._name_to_dim == result._name_to_dim
    assert union._obj == result._obj


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
@pytest.mark.parametrize("has_params", [True, False])
def test_set_intersection(ndims, has_params) -> None:
    a_dims, a_cond, b_dims, b_cond = _generate_dims_and_random_conditions(
        ndims, has_params)

    if has_params:
        a = nisl.make_set(f"[n] -> {{ [{a_dims}] : 0 <= {a_dims} < n }}")
        b = nisl.make_set(f"[m] -> {{ [{b_dims}] : 0 <= {b_dims} < m }}")
        result = nisl.make_set(
            f"""
            [n, m] -> {{ [{a_dims}, {b_dims}] : (0 <= {a_dims} < n) and (0 <= {b_dims} < m)}}
            """
        )
    else:
        a = nisl.make_set(f"{{ [{a_dims}] : {a_cond} }}")
        b = nisl.make_set(f"{{ [{b_dims}] : {b_cond} }}")
        result = nisl.make_set(
            f"""
            {{ [{a_dims}, {b_dims}] : ({a_cond}) and ({b_cond})}}
            """
        )

    intersection = a & b
    assert intersection._name_to_dim == result._name_to_dim
    assert intersection._obj == result._obj


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[0])
    else:
        from pytest import main
        main([__file__])
