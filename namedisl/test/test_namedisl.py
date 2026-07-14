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
from .utils_for_tests import generate_random_named_set
from namedisl import to_named
from namedisl.core import DimType


@pytest.mark.parametrize("ndims", [2, 3, 4, 5])
@pytest.mark.parametrize("has_params", [True, False])
def test_names(ndims: int, has_params: bool):
    s_param = "n" if has_params else None
    s, s_dims, _ = generate_random_named_set(ndims, "s", s_param)
    names = frozenset(s_dims.split(","))

    if s_param:
        names = names | frozenset({s_param})

    assert s.space.names == names


def test_public_dim_type_name_accessors() -> None:
    named_map = nisl.make_map("[n] -> { [i] -> [o] }")

    assert named_map.space.dim_names(DimType.in_) == {"i"}
    assert named_map.space.in_names == {"i"}
    assert named_map.space.dim_names(DimType.out) == {"o"}
    assert named_map.space.set_names == {"o"}
    assert named_map.space.dim_names(DimType.param) == {"n"}
    assert named_map.space.param_names == {"n"}


def test_public_dim_type_name_accessors_for_aff() -> None:
    named_aff = nisl.make_aff("[n] -> { [i] -> [i + n] }")

    assert named_aff.space.in_names == {"i"}
    assert named_aff.space.param_names == {"n"}


def test_add_set_and_parameter_names_reconstructs_expected_set() -> None:
    named_set = (
        nisl.make_set("{ [x] }")
        .add_dim_names(DimType.out, ["y"])
        .add_dim_names(DimType.param, ["p"])
    )
    expected = isl.Set("[p] -> { [y, x] }")
    assert named_set.equals(to_named(expected))


def test_add_output_input_and_parameter_names_reconstructs_expected_map() -> None:
    named_map = (
        nisl.make_map("{ [i] -> [o] }")
        .add_dim_names(DimType.out, ["o2"])
        .add_dim_names(DimType.in_, ["i2"])
        .add_dim_names(DimType.param, ["n"])
    )
    expected = isl.Map("[n] -> { [i2, i] -> [o2, o] }")

    assert named_map.equals(to_named(expected))


def test_add_input_and_parameter_names_reconstructs_expected_aff() -> None:
    named_aff = (
        nisl.make_aff("[n] -> { [i] -> [i + n] }")
        .add_dim_names(DimType.in_, ["j"])
        .add_dim_names(DimType.param, ["m"])
    )
    expected = isl.Aff("[m, n] -> { [j, i] -> [i + n] }")
    assert named_aff.equals(to_named(expected))


def test_add_dim_names_uses_dim_type() -> None:
    named_map = (
        nisl.make_map("{ [i] -> [o] }")
        .add_dim_names(DimType.in_, ["j"])
        .add_dim_names(DimType.param, ["p"])
        .add_dim_names(DimType.out, ["x"])
    )

    assert named_map.space.in_names == {"j", "i"}
    assert named_map.space.param_names == {"p"}
    assert named_map.space.set_names == {"x", "o"}


def test_move_dims_set() -> None:
    named_set = nisl.make_set("[p] -> { [x, y, z] : x + y = z and 0 <= x, y, z < p }")

    moved = named_set.move_dims(["z"], DimType.param)

    expected = isl.Set(
        "[p] -> { [x, y, z] : "
        "x + y = z and 0 <= x and 0 <= y and 0 <= z "
        "and x < p and y < p and z < p }"
    ).move_dims(
        isl.dim_type.param, 1,
        isl.dim_type.set, 2, 1
    )

    assert moved == to_named(expected)


def test_move_dims_map() -> None:
    named_map = nisl.make_map(
        "[p] -> { [i0, i1] -> [o0, o1, o2] : o0 = i0 and o1 = i1 and o2 = p }"
    )

    moved = named_map.move_dims(["o2"], DimType.in_)

    expected = isl.Map(
        "[p] -> { [i0, i1] -> [o0, o1, o2] : o0 = i0 and o1 = i1 and o2 = p }"
    ).move_dims(isl.dim_type.in_, 2, isl.dim_type.out, 2, 1)

    assert moved == to_named(expected)


def test_move_dims_multiple_names_preserves_relative_order() -> None:
    named_map = nisl.make_map(
        "[p] -> { [i0, i1] -> [o0, o1, o2] : o0 = i0 and o1 = i1 and o2 = p }"
    )

    moved = named_map.move_dims(["o1", "o2"], DimType.in_)

    expected = isl.Map(
        "[p] -> { [i0, i1] -> [o0, o1, o2] : o0 = i0 and o1 = i1 and o2 = p }"
    )
    expected = expected.move_dims(isl.dim_type.in_, 2, isl.dim_type.out, 1, 1)
    expected = expected.move_dims(isl.dim_type.in_, 3, isl.dim_type.out, 1, 1)

    assert moved == to_named(expected)


def test_rename_dims_set() -> None:
    named_set = nisl.make_set("[p] -> { [x, y] : x < p and y < p }")

    renamed = named_set.rename_dims({"x": "x_new", "p": "n"}.items())

    expected = isl.Set("[p] -> { [x, y] : x < p and y < p }")
    expected = expected.set_dim_name(isl.dim_type.set, 0, "x_new")
    expected = expected.set_dim_name(isl.dim_type.param, 0, "n")
    assert renamed == to_named(expected)


def test_rename_dims_map() -> None:
    named_map = nisl.make_map(
        "[p] -> { [i0, i1] -> [o0, o1] : o0 = i0 and o1 = p + i1 }"
    )

    renamed = named_map.rename_dims({"i1": "j", "o1": "x", "p": "n"}.items())

    expected = isl.Map(
        "[p] -> { [i0, i1] -> [o0, o1] : o0 = i0 and o1 = p + i1 }"
    )
    expected = expected.set_dim_name(isl.dim_type.in_, 1, "j")
    expected = expected.set_dim_name(isl.dim_type.out, 1, "x")
    expected = expected.set_dim_name(isl.dim_type.param, 0, "n")
    assert renamed == to_named(expected)


def test_rename_dims_rejects_renaming_to_existing_name() -> None:
    named_map = nisl.make_map("{ [i] -> [o] }")

    with pytest.raises(ValueError, match="existing name"):
        _ = named_map.rename_dims({"i": "o"}.items())


def test_rename_dims_rejects_unknown_name() -> None:
    named_set = nisl.make_set("{ [x] }")

    with pytest.raises(KeyError, match="y"):
        _ = named_set.rename_dims({"y": "z"}.items())


def test_duplicate_set_names_are_rejected() -> None:
    with pytest.raises(AssertionError):
        _ = nisl.make_set("{ [x, x] }")


def test_ticked_names_are_distinct_names() -> None:
    space = isl.Space.create_from_names(
        isl.DEFAULT_CONTEXT,
        set=["x", "x'"]
    )

    named_set = nisl.make_set(isl.Set.universe(space))

    assert named_set.space.names == frozenset({"x", "x'"})
