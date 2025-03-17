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


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
def test_align_two(ndims) -> None:
    dims_a = ",".join(f"a{d}" for d in range(ndims))
    dims_b = ",".join(f"b{d}" for d in range(ndims))
    a = nisl.make_basic_set(f"[n] -> {{ [{dims_a}] : 0 <= {dims_a} < n }}")
    b = nisl.make_basic_set(f"[n] -> {{ [{dims_b}] : 0 <= {dims_b} < n }}")

    a, b = nisl.align_two(a, b)
    assert a._name_to_dim == b._name_to_dim


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
def test_align_spaces(ndims) -> None:
    dims_a = ",".join(f"a{d}" for d in range(ndims))
    dims_b = ",".join(f"b{d}" for d in range(ndims))
    a = nisl.make_basic_set(f"[n] -> {{ [{dims_a}] : 0 <= {dims_a} < n }}")
    b = nisl.make_basic_set(f"[n] -> {{ [{dims_b}] : 0 <= {dims_b} < n }}")

    b = nisl.align_spaces(b, a)
    assert b.dim(nisl.dim_type.out) == 2*ndims


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
def test_basic_set_intersection(ndims) -> None:
    a_dims = ",".join(f"i{d}" for d in range(ndims))
    b_dims = ",".join(f"j{d}" for d in range(ndims))
    a = nisl.make_basic_set(f"[n] -> {{ [{a_dims}] : 0 <= {a_dims} < n }}")
    b = nisl.make_basic_set(f"[n] -> {{ [{b_dims}] : 0 <= {b_dims} < n }}")

    result = nisl.make_basic_set(
        f"""
        [n] ->
        {{ [{a_dims}, {b_dims}] : 0 <= {a_dims} < n and 0 <= {b_dims} < n}}
        """
    )

    assert (a & b) == result


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
def test_basic_map_intersection(ndims) -> None:
    a_in_dims = ",".join(f"i{d}" for d in range(ndims))
    a_out_dims = ",".join(f"j{d}" for d in range(ndims))

    b_in_dims = ",".join(f"a{d}" for d in range(ndims))
    b_out_dims = ",".join(f"b{d}" for d in range(ndims))

    a = nisl.make_basic_map(
        f"""
        [n] ->
        {{[{a_in_dims}] -> [{a_out_dims}]: 0 <= {a_in_dims}, {a_out_dims} < n}}
        """
    )
    b = nisl.make_basic_map(
        f"""
        [n] ->
        {{[{b_in_dims}] -> [{b_out_dims}]: 0 <= {b_in_dims}, {b_out_dims} < n}}
        """
    )

    domain_str = f"{a_in_dims},{b_in_dims}"
    range_str = f"{a_out_dims},{b_out_dims}"
    condition_str = f"0 <= {domain_str}, {range_str} < n"
    result = nisl.make_basic_map(
        f"[n] -> {{ [{domain_str}] -> [{range_str}] : {condition_str} }}"
    )

    assert (a & b) == result


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
def test_basic_set_union(ndims) -> None:
    a_dims = ",".join(f"i{d}" for d in range(ndims))
    b_dims = ",".join(f"j{d}" for d in range(ndims))
    a = nisl.make_basic_set(f"[n] -> {{ [{a_dims}] : 0 <= {a_dims} < n }}")
    b = nisl.make_basic_set(f"[n] -> {{ [{b_dims}] : 0 <= {b_dims} < n }}")

    result = nisl.make_set(
        f"""
        [n] ->
        {{ [{a_dims}, {b_dims}] : (0 <= {a_dims} < n) or (0 <= {b_dims} < n)}}
        """
    )

    assert (a | b) == result


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
def test_basic_map_union(ndims) -> None:
    a_in = ",".join(f"a_in{i}" for i in range(ndims))
    a_out = ",".join(f"a_out{i}" for i in range(ndims))
    a_cond = f"0 <= {a_in},{a_out} < n"
    a = nisl.make_basic_map(
        f"[n] -> {{ [{a_in}] -> [{a_out}] : {a_cond} }}"
    )

    b_in = ",".join(f"b_in{i}" for i in range(ndims))
    b_out = ",".join(f"b_out{i}" for i in range(ndims))
    b_cond = f"0 <= {b_in},{b_out} < n"
    b = nisl.make_basic_map(
        f"[n] -> {{ [{b_in}] -> [{b_out}] : 0 <= {b_in},{b_out} < n }}"
    )

    result = nisl.make_map(
        f"""
        [n] ->
        {{ [{a_in},{b_in}] -> [{a_out},{b_out}] : ({a_cond}) or ({b_cond})}}
        """
    )

    assert (a | b) == result


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
def test_set_union(ndims) -> None:
    a_dims = ",".join(f"i{d}" for d in range(ndims))
    b_dims = ",".join(f"j{d}" for d in range(ndims))
    a = nisl.make_set(f"[n] -> {{ [{a_dims}] : 0 <= {a_dims} < n }}")
    b = nisl.make_set(f"[n] -> {{ [{b_dims}] : 0 <= {b_dims} < n }}")

    result = nisl.make_set(
        f"""
        [n] ->
        {{ [{a_dims}, {b_dims}] : (0 <= {a_dims} < n) or (0 <= {b_dims} < n)}}
        """
    )

    assert (a | b) == result


@pytest.mark.parametrize("ndims", [1, 2, 4, 8])
def test_map_union(ndims) -> None:
    a_in = ",".join(f"a_in{i}" for i in range(ndims))
    a_out = ",".join(f"a_out{i}" for i in range(ndims))
    a_cond = f"0 <= {a_in},{a_out} < n"
    a = nisl.make_map(
        f"[n] -> {{ [{a_in}] -> [{a_out}] : {a_cond} }}"
    )

    b_in = ",".join(f"b_in{i}" for i in range(ndims))
    b_out = ",".join(f"b_out{i}" for i in range(ndims))
    b_cond = f"0 <= {b_in},{b_out} < n"
    b = nisl.make_map(
        f"[n] -> {{ [{b_in}] -> [{b_out}] : 0 <= {b_in},{b_out} < n }}"
    )

    result = nisl.make_map(
        f"""
        [n] ->
        {{ [{a_in},{b_in}] -> [{a_out},{b_out}] : ({a_cond}) or ({b_cond})}}
        """
    )

    assert (a | b) == result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[0])
    else:
        from pytest import main
        main([__file__])
