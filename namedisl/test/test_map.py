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

from utils_for_tests import generate_random_named_map


def test_map_from_str() -> None:
    m = nisl.make_map(
        "[n] -> { [i,j] -> [a,b] : 0 <= i, j < 10 and 0 <= a, b < 20 }")

    print(m._obj)
    print(m)


def test_map_from_map() -> None:
    m = isl.Map(
        "[n] -> { [i,j] -> [a,b] : 0 <= i, j < 10 and 0 <= a, b < 20 }")
    named_map = nisl.make_map(m)

    print(named_map._obj)
    print(named_map)


@pytest.mark.parametrize("ndims_domain", [2, 3, 4, 5])
@pytest.mark.parametrize("ndims_range",  [2, 3, 4, 5])
@pytest.mark.parametrize("has_params",   [True, False])
def test_map_equality(ndims_domain, ndims_range, has_params):
    if has_params:
        d_param = "n"
        r_param = "m"
    else:
        d_param = None
        r_param = None

    og_map, domain_info, range_info = generate_random_named_map(
        ndims_domain, "d", d_param,
        ndims_range, "r", r_param)

    _, d_dims, d_cond = domain_info
    _, r_dims, r_cond = range_info

    from itertools import permutations
    d_perms = list(permutations(d_dims.split(",")))
    r_perms = list(permutations(r_dims.split(",")))

    for d_perm, r_perm in zip(d_perms, r_perms):
        d_perm_dims = ",".join(p for p in d_perm)
        r_perm_dims = ",".join(p for p in r_perm)

        domain_str = f"{{ [{d_perm_dims}] : {d_cond} }}"
        range_str = f"{{ [{r_perm_dims}] : {r_cond} }}"

        if has_params:
            domain_str = f"[{d_param}] ->" + domain_str
            range_str = f"[{r_param}] ->" + range_str

        perm_map = nisl.make_map(
            isl.Map.from_domain_and_range(
                isl.Set(domain_str), isl.Set(range_str)
            )
        )

        assert perm_map == og_map


@pytest.mark.parametrize("ndims_domain", [2, 3, 4, 5])
@pytest.mark.parametrize("ndims_range",  [2, 3, 4, 5])
@pytest.mark.parametrize("has_params",   [True, False])
def test_map_union(ndims_domain, ndims_range, has_params):
    if has_params:
        d_param = "n"
        r_param = "m"
    else:
        d_param = None
        r_param = None

    x, x_domain_info, x_range_info = generate_random_named_map(
        ndims_domain, "x_in", d_param,
        ndims_range, "x_out", r_param
    )

    y, y_domain_info, y_range_info = generate_random_named_map(
        ndims_domain, "y_in", d_param,
        ndims_domain, "y_out", r_param
    )

    _, x_in_dims, x_in_cond = x_domain_info
    _, x_out_dims, x_out_cond = x_range_info

    _, y_in_dims, y_in_cond = y_domain_info
    _, y_out_dims, y_out_cond = y_range_info

    result_dims = f"[{x_in_dims}, {y_in_dims}] -> [{x_out_dims}, {y_out_dims}]"
    result_conds = f"({x_in_cond} and {x_out_cond}) or ({y_in_cond} and {y_out_cond})"

    result_str = "{" + result_dims + " : " + result_conds + "}"

    if has_params:
        result_str = f"[{d_param}, {r_param}] ->" + result_str

    result_map = nisl.make_map(result_str)

    assert (x | y) == result_map


@pytest.mark.parametrize("ndims_domain", [2, 3, 4, 5])
@pytest.mark.parametrize("ndims_range",  [2, 3, 4, 5])
@pytest.mark.parametrize("has_params",   [True, False])
def test_map_intersection(ndims_domain, ndims_range, has_params):
    if has_params:
        d_param = "n"
        r_param = "m"
    else:
        d_param = None
        r_param = None

    x, x_domain_info, x_range_info = generate_random_named_map(
        ndims_domain, "x_in", d_param,
        ndims_range, "x_out", r_param
    )

    y, y_domain_info, y_range_info = generate_random_named_map(
        ndims_domain, "y_in", d_param,
        ndims_domain, "y_out", r_param
    )

    _, x_in_dims, x_in_cond = x_domain_info
    _, x_out_dims, x_out_cond = x_range_info

    _, y_in_dims, y_in_cond = y_domain_info
    _, y_out_dims, y_out_cond = y_range_info

    result_dims = f"[{x_in_dims}, {y_in_dims}] -> [{x_out_dims}, {y_out_dims}]"
    result_conds = f"({x_in_cond} and {x_out_cond}) and ({y_in_cond} and {y_out_cond})"

    result_str = "{" + result_dims + " : " + result_conds + "}"

    if has_params:
        result_str = f"[{d_param}, {r_param}] ->" + result_str

    result_map = nisl.make_map(result_str)

    assert (x & y) == result_map


@pytest.mark.parametrize("ndims_domain", [1, 2, 4, 8])
@pytest.mark.parametrize("ndims_range", [1, 2, 4, 8])
def test_map_eliminate(ndims_domain, ndims_range):
    x, x_domain_info, x_range_info = generate_random_named_map(
        ndims_domain, "x_in", None,
        ndims_range, "x_out", None
    )

    _, x_in_dims, _ = x_domain_info
    _, x_out_dims, _ = x_range_info

    dims_to_remove = (x_in_dims + "," + x_out_dims).split(",")
    x = x.eliminate(dims_to_remove)

    assert x == nisl.make_map(f"{{[{x_in_dims}] -> [{x_out_dims}]}}")


@pytest.mark.parametrize("ndims_domain", [1, 2, 4, 8])
@pytest.mark.parametrize("ndims_range", [1, 2, 4, 8])
def test_map_project_out(ndims_domain, ndims_range):
    x, x_domain_info, x_range_info = generate_random_named_map(
        ndims_domain, "x_in", None,
        ndims_range, "x_out", None
    )

    _, x_in_dims, _ = x_domain_info
    _, x_out_dims, _ = x_range_info

    dims_to_remove = (x_in_dims + "," + x_out_dims).split(",")
    x = x.project_out(dims_to_remove)

    assert x == nisl.make_map("{[] -> []}")
