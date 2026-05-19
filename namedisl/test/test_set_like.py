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
from loopy.symbolic import pwaff_from_expr
from pymbolic import var
from typing_extensions import assert_type

import islpy as isl

import namedisl as nisl
from .utils_for_tests import generate_random_named_map, generate_random_named_set
from namedisl.core import _align_two


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


def test_set_add_constraint_uses_named_dimensions() -> None:
    set_ = nisl.make_set("{ [j, i] }")

    constrained = set_.add_constraint("i = j - 1")

    assert constrained == nisl.make_set("{ [j, i] : i = j - 1 }")


def test_set_add_constraint_accepts_multiple_constraints() -> None:
    set_ = nisl.make_set("{ [i, j, k] }")

    constrained = set_.add_constraint(["0 <= i", "j = i + 1", "k <= j"])

    assert constrained == nisl.make_set(
        "{ [i, j, k] : 0 <= i and j = i + 1 and k <= j }"
    )


def test_set_add_constraint_rejects_unknown_name() -> None:
    set_ = nisl.make_set("{ [i] }")

    with pytest.raises(ValueError, match="invalid constraint"):
        _ = set_.add_constraint("j = i")


def test_set_gist_simplifies_against_named_context() -> None:
    set_ = nisl.make_set(
        "{ [i, j, kb] : 0 <= i <= 13 and 0 <= j <= 13 "
        "and 0 <= kb <= 4 and kb <= 3 }"
    )
    context = nisl.make_set("{ [j, i, kb] : 0 <= i <= 13 and 0 <= j <= 13 }")

    assert set_.gist(context) == nisl.make_set("{ [i, j, kb] : 0 <= kb <= 3 }")


def test_basic_set_gist_preserves_basic_set_when_result_is_basic() -> None:
    set_ = nisl.make_basic_set("{ [i] : 0 <= i <= 10 }")
    context = nisl.make_set("{ [i] : i <= 5 or i >= 8 }")

    result = set_.gist(context)

    assert isinstance(result, nisl.BasicSet)
    assert result == nisl.make_basic_set("{ [i] : 0 <= i <= 10 }")


def test_set_subset_comparisons_align_by_name() -> None:
    smaller = nisl.make_set("{ [i] : 0 <= i < 5 }")
    larger = nisl.make_set("{ [j, i] : 0 <= i < 10 }")
    equal_reordered = nisl.make_set("{ [i, j] : 0 <= i < 5 }")

    assert smaller < larger
    assert smaller <= larger
    assert larger > smaller
    assert larger >= smaller
    assert smaller <= equal_reordered
    assert equal_reordered <= smaller
    assert not smaller < equal_reordered
    assert not larger <= smaller


def test_basic_set_subset_comparisons_allow_set_promotion() -> None:
    smaller = nisl.make_basic_set("{ [i] : 0 <= i < 5 }")
    larger = nisl.make_set("{ [j, i] : 0 <= i < 10 }")

    assert smaller < larger
    assert smaller <= larger
    assert larger > smaller
    assert larger >= smaller


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


def test_set_convex_hull_returns_basic_set() -> None:
    set_ = nisl.make_set(
        "{ [j, i] : "
        "(j = 0 and 0 <= i <= 2) or "
        "(j = 2 and 0 <= i <= 2) }"
    )

    result = set_.convex_hull()

    assert isinstance(result, nisl.BasicSet)
    assert result == nisl.make_basic_set(
        "{ [j, i] : 0 <= j <= 2 and 0 <= i <= 2 }"
    )


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

    # dim_{min,max} return raw isl.PwAff objects on a zero-dimensional set space.
    cond_pw_affs = [
        isl.PwAff(f"{{ [{cond.split('<')[2].strip(' ')}] }}")
        for cond in a_cond.split("and")
    ]

    for i, name in enumerate(a_dims.split(",")):
        assert a.dim_max(name) == (cond_pw_affs[i] - 1)


@pytest.mark.parametrize("ndims", [2, 4, 8])
def test_set_dim_min(ndims: int):
    a, a_dims, a_cond = generate_random_named_set(ndims, "a", None)

    # dim_{min,max} return raw isl.PwAff objects on a zero-dimensional set space.
    cond_pw_affs = [
        isl.PwAff(f"{{ [{cond.split('<')[0].strip(' ')}] }}")
        for cond in a_cond.split("and")
    ]

    for i, name in enumerate(a_dims.split(",")):
        assert a.dim_min(name) == cond_pw_affs[i]

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


def test_map_add_constraint_uses_input_output_and_parameter_names() -> None:
    map_ = nisl.make_map("[n] -> { [i] -> [j] }")

    constrained = map_.add_constraint("j = i + n")

    assert constrained == nisl.make_map("[n] -> { [i] -> [j] : j = i + n }")


def test_map_add_constraint_preserves_basic_map_type() -> None:
    map_ = nisl.make_basic_map("{ [i] -> [j] }")

    constrained = map_.add_constraint("j = i + 1")

    assert isinstance(constrained, nisl.BasicMap)
    assert constrained == nisl.make_basic_map("{ [i] -> [j] : j = i + 1 }")


def test_map_add_constraint_supports_previous_context_relation() -> None:
    relation = nisl.make_map(
        "{ [ki_prev, kb_prev] -> [ki_cur, kb_cur] : "
        "kb_cur = kb_prev + 1 }"
    )

    constrained = relation.add_constraint("ki_prev = ki_cur - 1")

    assert constrained == nisl.make_map(
        "{ [ki_prev, kb_prev] -> [ki_cur, kb_cur] : "
        "ki_prev = ki_cur - 1 and kb_cur = kb_prev + 1 }"
    )


def test_map_gist_simplifies_against_named_context() -> None:
    map_ = nisl.make_map(
        "{ [i, kb] -> [j] : 0 <= i <= 13 and 0 <= kb <= 4 "
        "and kb <= 3 and j = i }"
    )
    context = nisl.make_map("{ [kb, i] -> [j] : 0 <= i <= 13 and j = i }")

    assert map_.gist(context) == nisl.make_map("{ [i, kb] -> [j] : 0 <= kb <= 3 }")


def test_map_subset_comparisons_align_by_name() -> None:
    smaller = nisl.make_map("{ [i] -> [x] : x = i and 0 <= i < 5 }")
    larger = nisl.make_map("{ [j, i] -> [y, x] : x = i and 0 <= i < 10 }")
    equal_reordered = nisl.make_map(
        "{ [i, j] -> [x, y] : x = i and 0 <= i < 5 }"
    )

    assert smaller < larger
    assert smaller <= larger
    assert larger > smaller
    assert larger >= smaller
    assert smaller <= equal_reordered
    assert equal_reordered <= smaller
    assert not smaller < equal_reordered
    assert not larger <= smaller


def test_basic_map_subset_comparisons_allow_map_promotion() -> None:
    smaller = nisl.make_basic_map("{ [i] -> [x] : x = i and 0 <= i < 5 }")
    larger = nisl.make_map("{ [j, i] -> [y, x] : x = i and 0 <= i < 10 }")

    assert smaller < larger
    assert smaller <= larger
    assert larger > smaller
    assert larger >= smaller


def test_map_convex_hull_returns_basic_map() -> None:
    map_ = nisl.make_map(
        "{ [i] -> [j] : "
        "(i = 0 and j = 0) or "
        "(i = 2 and j = 2) }"
    )

    result = map_.convex_hull()

    assert isinstance(result, nisl.BasicMap)
    assert result == nisl.make_basic_map("{ [i] -> [j] : j = i and 0 <= i <= 2 }")


def test_subset_comparison_rejects_set_map_mismatch() -> None:
    set_ = nisl.make_set("{ [i] : 0 <= i < 5 }")
    map_ = nisl.make_map("{ [i] -> [x] : x = i and 0 <= i < 5 }")

    with pytest.raises(TypeError):
        _ = set_ <= map_


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
    assert isinstance(global_usage_map, nisl.Map)
    assert_type(global_usage_map, nisl.Map)
    compute_map = compute_map.rename_dims({
        "ii_s": "ii_s_out",
        "io": "io_out",
        "ki_s": "ki_s_out",
        "ko": "ko_out",
    })
    composed = global_usage_map.apply_range(compute_map)

    assert composed.input_names == frozenset(
        {"i", "ii", "ii_s", "io", "j", "ji", "ji_s", "jo",
         "k", "ki", "ki_s", "ko"}
    )
    assert composed.range().names == frozenset(
        {"ii_s_out", "io_out", "ki_s_out", "ko_out"}
    )


def test_map_apply_range_rejects_surviving_name_collisions() -> None:
    lhs = nisl.make_map("{ [x] -> [y] }")
    rhs = nisl.make_map("{ [y] -> [x] }")

    with pytest.raises(ValueError, match="duplicate surviving names"):
        _ = lhs.apply_range(rhs)


def test_map_apply_range_can_explicitly_rename_and_equate_collision() -> None:
    lhs = nisl.make_map("{ [x] -> [y] }")
    rhs = nisl.make_map("{ [y] -> [x] }").rename_dims({"x": "x_out"})

    result = lhs.apply_range(rhs).equate_dims("x", "x_out")

    assert result.input_names == frozenset({"x"})
    assert result.range().names == frozenset({"x_out"})
    assert (
        result.intersect_domain(nisl.make_set("{ [x] : x = 3 }"))
        .range()
        .dim_min("x_out")
        .plain_is_equal(isl.PwAff("{ [(3)] }"))
    )


def test_map_apply_range_can_equate_renamed_collisions_from_mapping() -> None:
    lhs = nisl.make_map("{ [x, z] -> [y] }")
    rhs = nisl.make_map("{ [y] -> [x, z] }").rename_dims({
        "x": "x_out",
        "z": "z_out",
    })

    result = lhs.apply_range(rhs).equate_dims({
        "x": "x_out",
        "z": "z_out",
    })

    assert result == nisl.make_map(
        "{ [x, z] -> [x_out, z_out] : x = x_out and z = z_out }"
    )


def test_equate_dims_mapping_rejects_unknown_name() -> None:
    map_ = nisl.make_map("{ [x] -> [x_out] }")

    with pytest.raises(ValueError, match="unknown name: missing"):
        _ = map_.equate_dims({"x": "missing"})


def test_map_apply_domain_rejects_surviving_name_collisions() -> None:
    lhs = nisl.make_map("{ [x] -> [y] }")
    rhs = nisl.make_map("{ [y] -> [x] }")

    with pytest.raises(ValueError, match="duplicate surviving names"):
        _ = rhs.apply_domain(lhs)


def test_map_apply_domain_can_explicitly_rename_and_equate_collision() -> None:
    lhs = nisl.make_map("{ [x] -> [y] }")
    rhs = nisl.make_map("{ [y] -> [x] }").rename_dims({"x": "x_out"})

    result = rhs.apply_domain(lhs).equate_dims("x", "x_out")

    assert result.input_names == frozenset({"x"})
    assert result.range().names == frozenset({"x_out"})
    assert (
        result.intersect_domain(nisl.make_set("{ [x] : x = 4 }"))
        .range()
        .dim_min("x_out")
        .plain_is_equal(isl.PwAff("{ [(4)] }"))
    )


def test_duplicate_map_names_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"duplicate|unnamed"):
        _ = nisl.make_map("{ [x] -> [x] }")


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

    assert m.as_pw_multi_aff() == m_isl.as_pw_multi_aff()


@pytest.mark.parametrize("ndims_domain", [1, 2, 4, 8])
@pytest.mark.parametrize("ndims_range", [1, 2, 4, 8])
def test_map_dim_max(ndims_domain: int, ndims_range: int):
    m, (_, in_names, in_conds), (_, out_names, out_conds) = generate_random_named_map(
        ndims_domain, "x_in", None,
        ndims_range, "x_out", None
    )

    # dim_{min,max} return raw isl.PwAff objects on a zero-dimensional set space.
    in_upper_bound_pw_maffs = [
        isl.PwAff(f"{{ [{int(cond.split('<')[2].strip(' '))}] }}")
        for cond in in_conds.split("and")
    ]

    for i, name in enumerate(in_names.split(",")):
        # NOTE: constructing PwAffs assumes starting index of 0, so subtract 1
        assert m.dim_max(name) == (in_upper_bound_pw_maffs[i] - 1)

    # dim_{min,max} return raw isl.PwAff objects on a zero-dimensional set space.
    out_upper_bound_pw_maffs = [
        isl.PwAff(f"{{ [{int(cond.split('<')[2].strip(' '))}] }}")
        for cond in out_conds.split("and")
    ]

    for i, name in enumerate(out_names.split(",")):
        # NOTE: constructing PwAffs assumes starting index of 0, so subtract 1
        assert m.dim_max(name) == (out_upper_bound_pw_maffs[i] - 1)


@pytest.mark.parametrize("ndims_domain", [1, 2, 4, 8])
@pytest.mark.parametrize("ndims_range", [1, 2, 4, 8])
def test_map_dim_min(ndims_domain: int, ndims_range: int):
    m, (_, in_names, in_conds), (_, out_names, out_conds) = generate_random_named_map(
        ndims_domain, "x_in", None,
        ndims_range, "x_out", None
    )

    # dim_{min,max} return raw isl.PwAff objects on a zero-dimensional set space.
    in_lower_bound_pw_maffs = [
        isl.PwAff(f"{{ [{int(cond.split('<')[0].strip(' '))}] }}")
        for cond in in_conds.split("and")
    ]

    for i, name in enumerate(in_names.split(",")):
        assert m.dim_min(name) == in_lower_bound_pw_maffs[i]

    # dim_{min,max} return raw isl.PwAff objects on a zero-dimensional set space.
    out_lower_bound_pw_maffs = [
        isl.PwAff(f"{{ [{int(cond.split('<')[0].strip(' '))}] }}")
        for cond in out_conds.split("and")
    ]

    for i, name in enumerate(out_names.split(",")):
        assert m.dim_min(name) == out_lower_bound_pw_maffs[i]

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
