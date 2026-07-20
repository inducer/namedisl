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
from .utils_for_tests import generate_random_named_map, generate_random_named_set
from namedisl.core import _find_joint_space


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

        assert a.equals(perm_set)


def test_set_like_equality_type_mismatch() -> None:
    set_ = nisl.make_set("{ [i] }")
    map_ = nisl.make_map("{ [i] -> [j] }")

    assert set_ != map_


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

    assert (a | b).equals(result)


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

    assert (a & b).equals(result)


def test_set_intersection_rejects_name_collision_across_dim_types() -> None:
    set_with_n = nisl.make_set("{ [n] }")
    param_with_n = nisl.make_set("[n] -> { [i] }")

    with pytest.raises(ValueError, match=r"duplicate|collision"):
        _find_joint_space(set_with_n.space, param_with_n.space)


def test_set_add_constraint_uses_named_dimensions() -> None:
    bset = nisl.make_basic_set("[m,n,p] -> { [j, i] }")

    v = bset.affs
    constrained = bset.add_constraint(
        nisl.Constraint.equality_from_aff(v["i"] - v["j"] + 1))

    assert constrained.as_set().equals(nisl.make_set("{ [j, i] : i = j - 1 }"))


def test_set_where() -> None:
    sp = nisl.Space.from_names(param=[], set=["i", "j", "k"])

    v = nisl.pw_affs_from_domain_space(sp)
    zero = v[0]
    i = v["i"]
    j = v["j"]
    k = v["k"]
    constrained = zero.where("<=", i) & j.where("=", i + 1) & k.where("<=", j)

    assert constrained == nisl.make_set(
        "{ [i, j, k] : 0 <= i and j = i + 1 and k <= j }"
    )


def test_set_gist_simplifies_against_named_context() -> None:
    set_ = nisl.make_set(
        "{ [i, j, kb] : 0 <= i <= 13 and 0 <= j <= 13 and 0 <= kb <= 4 and kb <= 3 }"
    )
    context = nisl.make_set("{ [j, i, kb] : 0 <= i <= 13 and 0 <= j <= 13 }")

    assert set_.gist(context) == nisl.make_set("{ [i, j, kb] : 0 <= kb <= 3 }")


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


def test_basic_set_intersection_promotion_to_set() -> None:
    basic = nisl.make_basic_set(
        "{ [ii_s, ji_s, k_s] : 0 <= ii_s <= 4 and 0 <= ji_s <= 4 and k_s = 0 }"
    )
    footprint = nisl.make_set(
        "{ [ii_s, ji_s, k_s] : "
        "(0 <= ii_s <= 4 and ji_s = 2 and k_s = 0) or "
        "(ii_s = 2 and 0 <= ji_s <= 4 and k_s = 0) }"
    )

    result = basic.as_set() & footprint

    assert isinstance(result, nisl.Set)
    reconstructed = result.as_isl()
    assert isinstance(reconstructed, isl.Set)
    assert reconstructed.n_basic_set() > 1


def test_set_convex_hull_returns_basic_set() -> None:
    set_ = nisl.make_set(
        "{ [j, i] : (j = 0 and 0 <= i <= 2) or (j = 2 and 0 <= i <= 2) }"
    )

    result = set_.convex_hull()

    assert isinstance(result, nisl.BasicSet)
    bs2 = nisl.make_basic_set("{ [j, i] : 0 <= j <= 2 and 0 <= i <= 2 }")
    print(result == bs2)
    if result != bs2:
        print(result == bs2)
        print(result.space.order_equals(bs2.space))
        print(result._obj)
        print(bs2._obj)
        print(result._obj.plain_is_equal(bs2._obj))
    assert result == bs2


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
def test_set_eliminate(ndims: int):
    a, a_dims, _ = generate_random_named_set(ndims, "a", None)
    a = a.eliminate(a_dims.split(","))

    assert a == nisl.make_set(f"{{[{a_dims}]}}")


def test_set_eliminate_rejects_unknown_name() -> None:
    set_ = nisl.make_set("{ [i] }")

    with pytest.raises(KeyError, match="missing"):
        _ = set_.eliminate(["missing"])


@pytest.mark.parametrize("ndims", [2, 4, 8])
def test_set_project_out(ndims: int):
    a, a_dims, _ = generate_random_named_set(ndims, "a", None)
    a = a.project_out(a_dims.split(","))

    assert a == nisl.make_set("{[]}")


def test_set_project_out_rejects_unknown_name() -> None:
    set_ = nisl.make_set("{ [i] }")

    with pytest.raises(KeyError, match="missing"):
        _ = set_.project_out(["missing"])


@pytest.mark.parametrize("ndims", [2, 4, 8])
def test_set_dim_max(ndims: int):
    a, a_dims, a_cond = generate_random_named_set(ndims, "a", None)

    # dim_{min,max} return raw isl.PwAff objects on a zero-dimensional set space.
    cond_pw_affs = [
        nisl.make_pw_aff(f"{{ [{cond.split('<')[2].strip(' ')}] }}")
        for cond in a_cond.split("and")
    ]

    for i, name in enumerate(a_dims.split(",")):
        assert a.dim_max(name) == (cond_pw_affs[i] - 1)


@pytest.mark.parametrize("ndims", [2, 4, 8])
def test_set_dim_min(ndims: int):
    a, a_dims, a_cond = generate_random_named_set(ndims, "a", None)

    # dim_{min,max} return raw isl.PwAff objects on a zero-dimensional set space.
    cond_pw_affs = [
        nisl.make_pw_aff(f"{{ [{cond.split('<')[0].strip(' ')}] }}")
        for cond in a_cond.split("and")
    ]

    for i, name in enumerate(a_dims.split(",")):
        assert a.dim_min(name) == cond_pw_affs[i]


def test_set_dim_bounds_reconstruct_parameter_metadata() -> None:
    set_ = nisl.make_set("[n] -> { [i] : 0 <= i < n }").rename_dims({
        "i": "j",
        "n": "m",
    }.items())

    assert set_.dim_min("j") == nisl.make_pw_aff("[m] -> { [(0)] : m > 0 }")
    assert set_.dim_max("j") == nisl.make_pw_aff("[m] -> { [(-1 + m)] : m > 0 }")


# }}}


# {{{ maps


def test_map_from_str() -> None:
    m = nisl.make_map("[n] -> { [i,j] -> [a,b] : 0 <= i, j < 10 and 0 <= a, b < 20 }")

    print(m._obj)
    print(m)


def test_map_from_map() -> None:
    m = isl.Map("[n] -> { [i,j] -> [a,b] : 0 <= i, j < 10 and 0 <= a, b < 20 }")
    named_map = nisl.make_map(m)

    print(named_map._obj)
    print(named_map)


def test_map_coalesce() -> None:
    map_ = nisl.make_map(
        "{ [i] -> [j = i] : 0 <= i < 5 or 5 <= i < 10 }"
    )
    assert len(map_.get_basic_maps()) == 2
    assert len(map_.coalesce().get_basic_maps()) == 1


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
        ndims_domain, "d", d_param, ndims_range, "r", r_param
    )

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
            isl.Map.from_domain_and_range(isl.Set(domain_str), isl.Set(range_str))
        )

        assert perm_map.equals(og_map)


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
        ndims_domain, "x_in", d_param, ndims_range, "x_out", r_param
    )

    y, y_domain_info, y_range_info = generate_random_named_map(
        ndims_domain, "y_in", d_param, ndims_domain, "y_out", r_param
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

    assert (x | y).equals(result_map)


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
        ndims_domain, "x_in", d_param, ndims_range, "x_out", r_param
    )

    y, y_domain_info, y_range_info = generate_random_named_map(
        ndims_domain, "y_in", d_param, ndims_domain, "y_out", r_param
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

    assert (x & y).equals(result_map)


def test_map_gist_simplifies_against_named_context() -> None:
    map_ = nisl.make_map(
        "{ [i, kb] -> [j] : 0 <= i <= 13 and 0 <= kb <= 4 and kb <= 3 and j = i }"
    )
    context = nisl.make_map("{ [kb, i] -> [j] : 0 <= i <= 13 and j = i }")

    assert map_.gist(context) == nisl.make_map("{ [i, kb] -> [j] : 0 <= kb <= 3 }")


def test_map_subset_comparisons_align_by_name() -> None:
    smaller = nisl.make_map("{ [i] -> [x] : x = i and 0 <= i < 5 }")
    larger = nisl.make_map("{ [j, i] -> [y, x] : x = i and 0 <= i < 10 }")
    equal_reordered = nisl.make_map("{ [i, j] -> [x, y] : x = i and 0 <= i < 5 }")

    assert smaller < larger
    assert smaller <= larger
    assert larger > smaller
    assert larger >= smaller
    assert smaller <= equal_reordered
    assert equal_reordered <= smaller
    assert not smaller < equal_reordered
    assert not larger <= smaller


def test_map_convex_hull_returns_basic_map() -> None:
    map_ = nisl.make_map("{ [i] -> [j] : (i = 0 and j = 0) or (i = 2 and j = 2) }")

    result = map_.convex_hull()

    assert isinstance(result, nisl.BasicMap)
    assert result == nisl.make_basic_map("{ [i] -> [j] : j = i and 0 <= i <= 2 }")


def test_subset_comparison_rejects_set_map_mismatch() -> None:
    set_ = nisl.make_set("{ [i] : 0 <= i < 5 }")
    map_ = nisl.make_map("{ [i] -> [x] : x = i and 0 <= i < 5 }")

    with pytest.raises(TypeError):
        _ = set_ <= map_  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_map_apply_range_rejects_surviving_name_collisions() -> None:
    lhs = nisl.make_map("{ [x] -> [y] }")
    rhs = nisl.make_map("{ [y] -> [x] }")

    with pytest.raises(ValueError, match="Uninvolved"):
        _ = lhs.apply_range(rhs)


def test_map_apply_range_can_explicitly_rename_and_equate_collision() -> None:
    lhs = nisl.make_map("{ [x] -> [y] }")
    rhs = nisl.make_map("{ [y] -> [x] }").rename_dims({"x": "x_out"}.items())

    result = lhs.apply_range(rhs).equate_dims([("x", "x_out")])

    assert result.space.in_names == frozenset({"x"})
    assert result.range().space.names == frozenset({"x_out"})
    assert (
        result
        .intersect_domain(nisl.make_set("{ [x] : x = 3 }"))
        .range()
        .dim_min("x_out")
        == nisl.make_pw_aff("{ [(3)] }")
    )


def test_map_apply_range_can_equate_renamed_collisions_from_mapping() -> None:
    lhs = nisl.make_map("{ [x, z] -> [y] }")
    rhs = nisl.make_map("{ [y] -> [x, z] }").rename_dims({
        "x": "x_out",
        "z": "z_out",
    }.items())

    result = lhs.apply_range(rhs).equate_dims([
        ("x", "x_out"),
        ("z", "z_out"),
    ])

    assert result == nisl.make_map(
        "{ [x, z] -> [x_out, z_out] : x = x_out and z = z_out }"
    )


def test_map_apply_range_unaligned_interface() -> None:
    lhs = nisl.make_map("{ [x, y] -> [b = x, a = y] }")
    rhs = nisl.make_map("{ [a, b] -> [u = a, v = b] }")

    ref = nisl.make_map("{ [x, y] -> [u = y, v = x] }")

    assert ref == lhs.apply_range(rhs)


def test_equate_dims_mapping_rejects_unknown_name() -> None:
    map_ = nisl.make_map("{ [x] -> [x_out] }")

    with pytest.raises(KeyError, match="missing"):
        _ = map_.equate_dims([("x", "missing")])


def test_map_apply_domain_rejects_surviving_name_collisions() -> None:
    lhs = nisl.make_map("{ [x] -> [y] }")
    rhs = nisl.make_map("{ [y] -> [x] }")

    with pytest.raises(ValueError, match="Uninvolved"):
        _ = rhs.apply_domain(lhs)


def test_map_apply_domain_can_explicitly_rename_and_equate_collision() -> None:
    lhs = nisl.make_map("{ [x] -> [y] }")
    rhs = nisl.make_map("{ [y] -> [x] }").rename_dims({"x": "x_out"}.items())

    result = rhs.apply_domain(lhs).equate_dims([("x", "x_out")])

    assert result.space.in_names == frozenset({"x"})
    assert result.range().space.names == frozenset({"x_out"})
    assert (
        result
        .intersect_domain(nisl.make_set("{ [x] : x = 4 }"))
        .range()
        .dim_min("x_out")
        == nisl.make_pw_aff("{ [(4)] }")
    )


def test_map_apply_domain_unaligned_interface() -> None:
    lhs = nisl.make_map("{ [b, a] -> [u = a, v = b] }")
    rhs = nisl.make_map("{ [x, y] -> [a = x, b = y] }")

    ref = nisl.make_map("{ [x, y] -> [u = x, v = y] }")

    assert ref == lhs.apply_domain(rhs)


def test_duplicate_map_names_are_rejected() -> None:
    with pytest.raises(AssertionError):
        _ = nisl.make_map("{ [x] -> [x] }")


@pytest.mark.parametrize("ndims_domain", [1, 2, 4, 8])
@pytest.mark.parametrize("ndims_range", [1, 2, 4, 8])
def test_map_eliminate(ndims_domain: int, ndims_range: int):
    x, x_domain_info, x_range_info = generate_random_named_map(
        ndims_domain, "x_in", None, ndims_range, "x_out", None
    )

    _, x_in_dims, _ = x_domain_info
    _, x_out_dims, _ = x_range_info

    dims_to_remove = (x_in_dims + "," + x_out_dims).split(",")
    x = x.eliminate(dims_to_remove)

    assert x == nisl.make_map(f"{{[{x_in_dims}] -> [{x_out_dims}]}}")


def test_map_eliminate_rejects_unknown_name() -> None:
    map_ = nisl.make_map("{ [i] -> [j] }")

    with pytest.raises(KeyError, match="missing"):
        _ = map_.eliminate(["missing"])


@pytest.mark.parametrize("ndims_domain", [1, 2, 4, 8])
@pytest.mark.parametrize("ndims_range", [1, 2, 4, 8])
def test_map_project_out(ndims_domain: int, ndims_range: int):
    x, x_domain_info, x_range_info = generate_random_named_map(
        ndims_domain, "x_in", None, ndims_range, "x_out", None
    )

    _, x_in_dims, _ = x_domain_info
    _, x_out_dims, _ = x_range_info

    dims_to_remove = (x_in_dims + "," + x_out_dims).split(",")
    x = x.project_out(dims_to_remove)

    assert x == nisl.make_map("{[] -> []}")


def test_map_project_out_rejects_unknown_name() -> None:
    map_ = nisl.make_map("{ [i] -> [j] }")

    with pytest.raises(KeyError, match="missing"):
        _ = map_.project_out(["missing"])


def test_map_as_pw_multi_aff():
    spec = "{ [i] -> [io, ii] : i = 32 * io + ii and 0 <= ii < 32 }"
    m = nisl.make_map(spec)
    isl.Map(spec)

    o1 = m.as_pw_multi_aff()
    o2 = nisl.make_pw_multi_aff("{ [i] -> [io = (floor((i)/32)), ii = ((i) mod 32)] }")
    assert o1 == o2


def test_map_domain_produces_suitable_space() -> None:
    m = nisl.make_map("{ [x] -> [y] : y = x + 1 }")
    m.domain()

# }}}


# {{{ basic{map, set}

def test_basic_map_from_str() -> None:
    m = nisl.make_basic_map(
        "[n] -> { [i,j] -> [a,b] : 0 <= i, j < 10 and 0 <= a, b < 20 }"
    )

    print(m._obj)
    print(m)


def test_basic_map_from_map() -> None:
    m = isl.BasicMap("[n] -> { [i,j] -> [a,b] : 0 <= i, j < 10 and 0 <= a, b < 20 }")
    named_map = nisl.make_basic_map(m)

    print(named_map._obj)
    print(named_map)


def test_basic_map_domain_produces_suitable_space() -> None:
    m = nisl.make_basic_map("{ [x] -> [y] : y = x + 1 }")
    m.domain()


# }}}
