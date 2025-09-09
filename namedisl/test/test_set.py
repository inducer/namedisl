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

import islpy as isl
import namedisl as nisl

from utils_for_tests import generate_random_named_set


def test_set_from_str() -> None:
    s = nisl.make_set("[n] -> { [i]: 0 <= i < n }")
    
    print("Original map:", s)
    print("Internal representation:", s._obj)
    print(s)


def test_set_from_set() -> None:
    s = isl.Set("[n] -> { [i, j] : 0 <= i, j < n }")
    named_set = nisl.make_set(s)

    print("Original map:", named_set)
    print("Internal representation:", named_set._obj)
    print(named_set)


@pytest.mark.parametrize("ndims",      [2, 3, 4, 5])
@pytest.mark.parametrize("has_params", [True, False])
def test_set_equality(ndims, has_params):
    if has_params:
        a_param = "n"
    else:
        a_param = None

    a, a_dims, a_cond = generate_random_named_set(ndims, "a", a_param)

    from itertools import permutations
    for perm in list(permutations(a_dims.split(","))):
        perm_dims = ",".join(p for p in perm)
        set_str = f"{{ [{perm_dims}] : {a_cond} }}"
        if has_params:
            set_str = f"[{a_param}] ->" + set_str
        perm_set = nisl.make_set(set_str)

        assert a == perm_set


@pytest.mark.parametrize("ndims",      [1, 2, 4, 8])
@pytest.mark.parametrize("has_params", [True, False])
def test_set_union(ndims, has_params):

    if has_params:
        a_param = "n"
        b_param = "m"
    else:
        a_param = None
        b_param = None

    a, a_dims, a_cond = generate_random_named_set(ndims, "a", a_param)
    b, b_dims, b_cond = generate_random_named_set(ndims, "b", b_param)

    set_str = f"{{ [{a_dims}, {b_dims}] : ({a_cond}) or ({b_cond})}}"
    if has_params:
        set_str = "[n, m] -> " + set_str 

    result = nisl.make_set(set_str)

    assert (a | b) == result


@pytest.mark.parametrize("ndims",      [1, 2, 4, 8])
@pytest.mark.parametrize("has_params", [True, False])
def test_set_intersection(ndims, has_params):

    if has_params:
        a_param = "n"
        b_param = "m"
    else:
        a_param = None
        b_param = None

    a, a_dims, a_cond = generate_random_named_set(ndims, "a", a_param)
    b, b_dims, b_cond = generate_random_named_set(ndims, "b", b_param)

    set_str = f"{{ [{a_dims}, {b_dims}] : ({a_cond}) and ({b_cond})}}"
    if has_params:
        set_str = "[n, m] -> " + set_str 

    result = nisl.make_set(set_str)

    assert (a & b) == result


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
def test_eliminate(ndims):
    a, a_dims, _ = generate_random_named_set(ndims, "a", None)

    for name in a_dims.split(","):
        a = a.eliminate(name) 

    assert a == nisl.make_set(f"{{[{a_dims}]}}")


@pytest.mark.parametrize("ndims", [2, 4, 8])
def test_project_out(ndims):
    a, a_dims, a_conds = generate_random_named_set(ndims, "a", None)

    # keep at least one name around so ISL doesn't complain
    from random import randint 
    a_dims = a_dims.split(",")
    index_of_name_to_keep = randint(0, len(a_dims)-1) 
    kept_name = a_dims[index_of_name_to_keep]
    kept_cond = a_conds.split("and")[index_of_name_to_keep]

    dims_to_remove = [dim for dim in a_dims if dim != kept_name]
    a = a.project_out(dims_to_remove)

    assert a == nisl.make_set(f"{{ [{kept_name}] : {kept_cond} }}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[0])
    else:
        from pytest import main
        main([__file__])
