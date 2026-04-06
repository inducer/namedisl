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
from namedisl.core import _align_two
from loopy.symbolic import pwaff_from_expr
from pymbolic import var
from .utils_for_tests import generate_random_named_map, generate_random_named_set


# {{{ sets

def test_set_from_str() -> None:
    s = nisl.make_set("[n] -> { [i]: 0 <= i < n }")

    print(s._obj)
    print(s)


def test_set_from_set() -> None:
    s = isl.Set("[n] -> { [i, j] : 0 <= i, j < n }")
    named_set = nisl.make_set(s)

    print(named_set._obj)
    print(named_set)


@pytest.mark.parametrize("ndims", [2, 3, 4, 5])
@pytest.mark.parametrize("has_params", [True, False])
def test_set_equality(ndims: int, has_params: bool):
    a_param = "n" if has_params else None

    a, a_dims, a_cond = generate_random_named_set(ndims, "a", a_param)

    from itertools import permutations
    for perm in list(permutations(a_dims.split(","))):
        perm_dims = ",".join(p for p in perm)
        set_str = f"{{ [{perm_dims}] : {a_cond} }}"
        if has_params:
            set_str = f"[{a_param}] ->" + set_str
        perm_set = nisl.make_set(set_str)

        assert a == perm_set


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
@pytest.mark.parametrize("has_params", [True, False])
def test_set_union(ndims: int, has_params: bool):

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


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
@pytest.mark.parametrize("has_params", [True, False])
def test_set_intersection(ndims: int, has_params: bool):

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


def test_basic_set_intersection_promotes_to_set() -> None:
    basic = nisl.make_basic_set(
        "{ [ii_s, ji_s, k_s] : 0 <= ii_s <= 4 and 0 <= ji_s <= 4 and k_s = 0 }"
    )
    footprint = nisl.make_set(
        "{ [ii_s, ji_s, k_s] : "
        "(0 <= ii_s <= 4 and ji_s = 2 and k_s = 0) or "
        "(ii_s = 2 and 0 <= ji_s <= 4 and k_s = 0) }"
    )

    result = basic & footprint

    assert isinstance(result, nisl.Set)
    reconstructed = result._reconstruct_isl_object()
    assert isinstance(reconstructed, isl.Set)
    assert reconstructed.n_basic_set() > 1


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
def test_set_eliminate(ndims: int):
    a, a_dims, _ = generate_random_named_set(ndims, "a", None)
    a = a.eliminate(a_dims.split(","))

    assert a == nisl.make_set(f"{{[{a_dims}]}}")


@pytest.mark.parametrize("ndims", [2, 4, 8])
def test_set_project_out(ndims: int):
    a, a_dims, _ = generate_random_named_set(ndims, "a", None)
    a = a.project_out(a_dims.split(","))

    assert a == nisl.make_set("{[]}")


@pytest.mark.parametrize("ndims", [2, 4, 8])
def test_set_dim_max(ndims: int):
    a, a_dims, a_cond = generate_random_named_set(ndims, "a", None)

    # unnamed, so use isl.PwAff instead of nisl.make_pw_aff
    cond_pw_affs = [
        isl.PwAff(f"{{ [] -> [{cond.split('<')[2].strip(' ')}] }}")
        for cond in a_cond.split("and")
    ]

    for i, name in enumerate(a_dims.split(",")):
        assert a.dim_max(name)._reconstruct_isl_object() == (cond_pw_affs[i] - 1)


@pytest.mark.parametrize("ndims", [2, 4, 8])
def test_set_dim_min(ndims: int):
    a, a_dims, a_cond = generate_random_named_set(ndims, "a", None)

    # unnamed, so use isl.PwAff instead of nisl.make_pw_aff
    cond_pw_affs = [
        isl.PwAff(f"{{ [] -> [{cond.split('<')[0].strip(' ')}] }}")
        for cond in a_cond.split("and")
    ]

    for i, name in enumerate(a_dims.split(",")):
        assert a.dim_min(name)._reconstruct_isl_object() == cond_pw_affs[i]

# }}}


# {{{ maps

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
@pytest.mark.parametrize("ndims_range", [2, 3, 4, 5])
@pytest.mark.parametrize("has_params", [True, False])
def test_map_equality(ndims_domain: int, ndims_range: int, has_params: bool):
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

    for d_perm, r_perm in zip(d_perms, r_perms, strict=False):
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
@pytest.mark.parametrize("ndims_range", [2, 3, 4, 5])
@pytest.mark.parametrize("has_params", [True, False])
def test_map_union(ndims_domain: int, ndims_range: int, has_params: bool):
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
@pytest.mark.parametrize("ndims_range", [2, 3, 4, 5])
@pytest.mark.parametrize("has_params", [True, False])
def test_map_intersection(ndims_domain: int, ndims_range: int, has_params: bool):
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


def test_map_alignment_syncs_internal_output_positions_and_names() -> None:
    lhs = nisl.make_map("{ [i] -> [x] }")
    rhs = nisl.make_map("{ [i] -> [y, x] }")

    aligned_lhs, aligned_rhs = _align_two(lhs, rhs)

    lhs_names = [
        aligned_lhs._obj.get_dim_name(isl.dim_type.set, dim)
        for dim in range(aligned_lhs._obj.dim(isl.dim_type.set))
    ]
    rhs_names = [
        aligned_rhs._obj.get_dim_name(isl.dim_type.set, dim)
        for dim in range(aligned_rhs._obj.dim(isl.dim_type.set))
    ]

    assert lhs_names == ["x", "y", "i"]
    assert rhs_names == ["x", "y", "i"]


def test_map_alignment_syncs_internal_input_and_parameter_positions_and_names() -> None:
    lhs = nisl.make_map("[n] -> { [i] -> [x] }")
    rhs = nisl.make_map("[m, n] -> { [j, i] -> [x] }")

    aligned_lhs, aligned_rhs = _align_two(lhs, rhs)

    lhs_names = [
        aligned_lhs._obj.get_dim_name(isl.dim_type.set, dim)
        for dim in range(aligned_lhs._obj.dim(isl.dim_type.set))
    ]
    rhs_names = [
        aligned_rhs._obj.get_dim_name(isl.dim_type.set, dim)
        for dim in range(aligned_rhs._obj.dim(isl.dim_type.set))
    ]

    assert lhs_names == ["x", "i", "j", "m", "n"]
    assert rhs_names == ["x", "i", "j", "m", "n"]


def test_map_apply_range_for_compute_a_tile_usage_map() -> None:
    bm = 32
    bk = 16

    compute_map = nisl.make_map(f"""{{
        [is, ks] -> [ii_s, io, ki_s, ko] :
            is = io * {bm} + ii_s and
            ks = ko * {bk} + ki_s
    }}""")

    usage_domain = nisl.make_set(
        "{ [i, j, k, io, jo, ko, ii, ji, ki, ii_s, ji_s, ki_s] }"
    )
    global_usage_map = nisl.make_map_from_domain_and_range(
        usage_domain,
        nisl.make_set("{ [is, ks] }")
    )

    local_usage_mpwaff = isl.MultiPwAff.zero(global_usage_map.get_space())
    for idx, expr in enumerate([var("i"), var("k")]):
        local_space = local_usage_mpwaff.get_at(idx).get_space().domain()
        local_usage_mpwaff = local_usage_mpwaff.set_pw_aff(
            idx,
            pwaff_from_expr(local_space, expr)
        )

    local_usage_map = nisl.make_map(local_usage_mpwaff.as_map())
    local_usage_map = local_usage_map.intersect_domain(
        nisl.make_basic_set(
            "{ [i, j, k, io, jo, ko, ii, ji, ki, ii_s, ji_s, ki_s] }"
        )
    )

    global_usage_map = global_usage_map | local_usage_map
    composed = global_usage_map.apply_range(compute_map)

    assert frozenset(
        name.rstrip("'") for name in composed.input_names
    ) == frozenset(
        {"i", "ii", "ii_s", "io", "j", "ji", "ji_s", "jo",
         "k", "ki", "ki_s", "ko"}
    )
    assert composed.range() == nisl.make_set("{ [ii_s, io, ki_s, ko] }")


def test_map_apply_domain_accepts_logically_equal_ticked_interface_names() -> None:
    lhs = nisl.make_map("{ [x] -> [y] }").apply_range(
        nisl.make_map("{ [y] -> [x] }")
    )
    rhs = nisl.make_map("{ [p] -> [x] }")

    result = lhs.apply_domain(rhs)

    assert result.range() == nisl.make_set("{ [x] }")


def test_map_domain_canonicalizes_single_remaining_ticked_name() -> None:
    m = nisl.make_map("{ [x] -> [y] }").apply_range(
        nisl.make_map("{ [y] -> [x] }")
    )

    domain = m.domain()

    assert domain == nisl.make_set("{ [x] }")
    assert domain.names == frozenset({"x"})


def test_map_empty_from_space_preserves_names_and_is_empty() -> None:
    space = isl.Space.create_from_names(
        isl.DEFAULT_CONTEXT,
        params=["n"],
        in_=["i", "j"],
        out=["x", "y"]
    )

    m = nisl.Map.empty(space)

    assert m._reconstruct_isl_object().is_empty()
    assert m.input_names == frozenset({"i", "j"})
    assert m.range() == nisl.make_set("[n] -> { [x, y] : false }")


def test_basic_map_empty_from_space_preserves_names_and_is_empty() -> None:
    space = isl.Space.create_from_names(
        isl.DEFAULT_CONTEXT,
        in_=["i"],
        out=["x"]
    )

    m = nisl.BasicMap.empty(space)

    assert m._reconstruct_isl_object().is_empty()
    assert m.input_names == frozenset({"i"})
    assert m.range() == nisl.make_basic_set("{ [x] : 1 = 0 }")


def test_map_empty_matches_existing_named_space() -> None:
    template = nisl.make_map("[n] -> { [i, k] -> [ii_s, io, ki_s, ko] }")

    empty_map = nisl.Map.empty(template.get_space())

    assert empty_map._reconstruct_isl_object().is_empty()
    assert empty_map.input_names == template.input_names
    assert empty_map.range()._reconstruct_isl_object().is_empty()
    assert empty_map.range().names == template.range().names


def test_empty_map_is_identity_for_union() -> None:
    space = isl.Space.create_from_names(
        isl.DEFAULT_CONTEXT,
        in_=["i"],
        out=["x"]
    )
    empty_map = nisl.Map.empty(space)
    nonempty_map = nisl.make_map("{ [i] -> [x] }")

    assert (empty_map | nonempty_map) == nonempty_map
    assert (nonempty_map | empty_map) == nonempty_map


@pytest.mark.parametrize("ndims_domain", [1, 2, 4, 8])
@pytest.mark.parametrize("ndims_range", [1, 2, 4, 8])
def test_map_eliminate(ndims_domain: int, ndims_range: int):
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
def test_map_project_out(ndims_domain: int, ndims_range: int):
    x, x_domain_info, x_range_info = generate_random_named_map(
        ndims_domain, "x_in", None,
        ndims_range, "x_out", None
    )

    _, x_in_dims, _ = x_domain_info
    _, x_out_dims, _ = x_range_info

    dims_to_remove = (x_in_dims + "," + x_out_dims).split(",")
    x = x.project_out(dims_to_remove)

    assert x == nisl.make_map("{[] -> []}")


def test_map_as_pw_multi_aff():
    spec = "{ [i] -> [io, ii] : i = 32 * io + ii and 0 <= ii < 32 }"
    m = nisl.make_map(spec)
    m_isl = isl.Map(spec)

    assert m.as_pw_multi_aff()._reconstruct_isl_object() == m_isl.as_pw_multi_aff()


@pytest.mark.parametrize("ndims_domain", [1, 2, 4, 8])
@pytest.mark.parametrize("ndims_range", [1, 2, 4, 8])
def test_map_dim_max(ndims_domain: int, ndims_range: int):
    m, (_, in_names, in_conds), (_, out_names, out_conds) = generate_random_named_map(
        ndims_domain, "x_in", None,
        ndims_range, "x_out", None
    )

    # unnamed, so use isl.PwAff instead of nisl.make_pw_aff
    in_upper_bound_pw_maffs = [
        isl.PwAff(f"{{ [] -> [{int(cond.split('<')[2].strip(' '))}] }}")
        for cond in in_conds.split("and")
    ]

    for i, name in enumerate(in_names.split(",")):
        # NOTE: constructing PwAffs assumes starting index of 0, so subtract 1
        assert m.dim_max(name)._obj == (in_upper_bound_pw_maffs[i] - 1)

    # unnamed, so use isl.PwAff instead of nisl.make_pw_aff
    out_upper_bound_pw_maffs = [
        isl.PwAff(f"{{ [] -> [{int(cond.split('<')[2].strip(' '))}] }}")
        for cond in out_conds.split("and")
    ]

    for i, name in enumerate(out_names.split(",")):
        # NOTE: constructing PwAffs assumes starting index of 0, so subtract 1
        assert m.dim_max(name)._obj == (out_upper_bound_pw_maffs[i] - 1)


@pytest.mark.parametrize("ndims_domain", [1, 2, 4, 8])
@pytest.mark.parametrize("ndims_range", [1, 2, 4, 8])
def test_map_dim_min(ndims_domain: int, ndims_range: int):
    m, (_, in_names, in_conds), (_, out_names, out_conds) = generate_random_named_map(
        ndims_domain, "x_in", None,
        ndims_range, "x_out", None
    )

    # unnamed, so use isl.PwAff instead of nisl.make_pw_aff
    in_lower_bound_pw_maffs = [
        isl.PwAff(f"{{ [] -> [{int(cond.split('<')[0].strip(' '))}] }}")
        for cond in in_conds.split("and")
    ]

    for i, name in enumerate(in_names.split(",")):
        assert m.dim_min(name)._obj == in_lower_bound_pw_maffs[i]

    # unnamed, so use isl.PwAff instead of nisl.make_pw_aff
    out_lower_bound_pw_maffs = [
        isl.PwAff(f"{{ [] -> [{int(cond.split('<')[0].strip(' '))}] }}")
        for cond in out_conds.split("and")
    ]

    for i, name in enumerate(out_names.split(",")):
        assert m.dim_min(name)._obj == out_lower_bound_pw_maffs[i]

# }}}


# {{{ basic{map, set}

def test_basic_map_from_str() -> None:
    m = nisl.make_basic_map(
        "[n] -> { [i,j] -> [a,b] : 0 <= i, j < 10 and 0 <= a, b < 20 }")

    print(m._obj)
    print(m)


def test_basic_map_from_map() -> None:
    m = isl.BasicMap(
        "[n] -> { [i,j] -> [a,b] : 0 <= i, j < 10 and 0 <= a, b < 20 }")
    named_map = nisl.make_basic_map(m)

    print(named_map._obj)
    print(named_map)

# }}}
