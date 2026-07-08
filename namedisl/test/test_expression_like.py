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
from constantdict import constantdict

import islpy as isl

import namedisl as nisl


ScalarIslExpression = isl.Aff | isl.PwAff | isl.QPolynomial | isl.PwQPolynomial


def _is_zero_expression(expr: ScalarIslExpression) -> bool:
    if isinstance(expr, isl.Aff):
        return bool(expr.plain_is_zero())
    if isinstance(expr, isl.PwAff):
        return bool(expr.plain_is_equal(expr * 0))
    return bool(expr.is_zero())


# {{{ affs

def test_aff_from_str():
    spec = "[n] -> { [i] -> [2 * i + n] }"
    named_aff = nisl.make_aff(spec)
    aff = isl.Aff(spec)

    print(named_aff)
    print(aff)

    assert aff == named_aff.as_isl()


def test_aff_from_aff():
    aff = isl.Aff("[n] -> { [i] -> [2 * i + n] }")
    named_aff = nisl.make_aff(aff)

    print(named_aff)
    print(aff)

    assert aff == named_aff.as_isl()


def test_aff_binary_ops():
    spec = "[n] -> { [i] -> [2 * i + n] }"
    named_aff = nisl.make_aff(spec)
    aff = isl.Aff(spec)

    aff_p_aff = aff + aff
    aff_p_1 = aff + 1

    naff_p_naff = named_aff + named_aff
    naff_p_1 = named_aff + 1

    assert naff_p_naff.as_isl() == aff_p_aff
    assert naff_p_1.as_isl() == aff_p_1

    aff_s_aff = aff - aff
    aff_s_1 = aff - 1

    naff_s_naff = named_aff - named_aff
    naff_s_1 = named_aff - 1

    assert naff_s_naff.as_isl() == aff_s_aff
    assert naff_s_1.as_isl() == aff_s_1

    aff_m_3 = aff * 3
    naff_m_3 = named_aff * 3

    assert naff_m_3.as_isl() == aff_m_3

# }}}


# {{{ pwaffs

def test_pwaff_from_str():
    spec = "[n] -> { [i] -> [2 * i + n] }"
    named_pwaff = nisl.make_pw_aff(spec)
    pwaff = isl.PwAff(spec)

    print(named_pwaff)
    print(pwaff)

    assert pwaff == named_pwaff.as_isl()


def test_pwaff_from_pwaff():
    pwaff = isl.PwAff("[n] -> { [i] -> [2 * i + n] }")
    named_pwaff = nisl.make_pw_aff(pwaff)

    print(named_pwaff)
    print(pwaff)

    assert pwaff == named_pwaff.as_isl()


def test_pwaff_binary_ops():
    spec = "[n] -> { [i] -> [2 * i + n] }"
    named_pwaff = nisl.make_pw_aff(spec)
    pwaff = isl.PwAff(spec)

    pwaff_p_pwaff = pwaff + pwaff
    pwaff_p_1 = pwaff + 1

    npwaff_p_npwaff = named_pwaff + named_pwaff
    npwaff_p_1 = named_pwaff + 1

    assert npwaff_p_npwaff.as_isl() == pwaff_p_pwaff
    assert npwaff_p_1.as_isl() == pwaff_p_1

    pwaff_s_pwaff = pwaff - pwaff
    pwaff_s_1 = pwaff - 1

    npwaff_s_npwaff = named_pwaff - named_pwaff
    npwaff_s_1 = named_pwaff - 1

    assert npwaff_s_npwaff.as_isl() == pwaff_s_pwaff
    assert npwaff_s_1.as_isl() == pwaff_s_1

    pwaff_m_3 = pwaff * 3
    npwaff_m_3 = named_pwaff * 3

    assert npwaff_m_3.as_isl() == pwaff_m_3


def test_mixed_aff_and_pwaff_binary_op_promotes_to_pwaff() -> None:
    aff = nisl.make_aff("{ [i] -> [i] }")
    pwaff = nisl.make_pw_aff("{ [i] -> [i] }")

    result = aff + pwaff

    assert isinstance(result, nisl.PwAff)
    assert result.as_isl() == (
        aff.as_isl().to_pw_aff() + pwaff.as_isl()
    )


def test_expression_equality_type_mismatch_raises_not_implemented_error() -> None:
    aff = nisl.make_aff("{ [i] -> [i] }")
    pwaff = nisl.make_pw_aff("{ [i] -> [i] }")

    with pytest.raises(NotImplementedError, match="Objects are not of the same type"):
        _ = aff == pwaff


def test_reflected_integer_expression_ops() -> None:
    aff_expr = nisl.make_aff("{ [i] -> [i] }")
    aff_obj = aff_expr.as_isl()
    assert _is_zero_expression((1 + aff_expr).as_isl() - (1 + aff_obj))
    assert _is_zero_expression((1 - aff_expr).as_isl() - (1 - aff_obj))
    assert _is_zero_expression((2 * aff_expr).as_isl() - (2 * aff_obj))

    pw_aff_expr = nisl.make_pw_aff("{ [i] -> [i] }")
    pw_aff_obj = pw_aff_expr.as_isl()
    assert _is_zero_expression(
        (1 + pw_aff_expr).as_isl() - (1 + pw_aff_obj)
    )
    assert _is_zero_expression(
        (1 - pw_aff_expr).as_isl() - (1 - pw_aff_obj)
    )
    assert _is_zero_expression(
        (2 * pw_aff_expr).as_isl() - (2 * pw_aff_obj)
    )

    qpoly_expr = nisl.make_qpolynomial("{ [i] -> i }")
    qpoly_obj = qpoly_expr.as_isl()
    assert _is_zero_expression(
        (1 + qpoly_expr).as_isl() - (1 + qpoly_obj)
    )
    assert _is_zero_expression(
        (1 - qpoly_expr).as_isl() - (1 - qpoly_obj)
    )
    assert _is_zero_expression(
        (2 * qpoly_expr).as_isl() - (2 * qpoly_obj)
    )

    pw_qpoly_expr = nisl.make_pw_qpolynomial("{ [i] -> i }")
    pw_qpoly_obj = pw_qpoly_expr.as_isl()
    assert _is_zero_expression(
        (1 + pw_qpoly_expr).as_isl() - (1 + pw_qpoly_obj)
    )
    assert _is_zero_expression(
        (1 - pw_qpoly_expr).as_isl() - (1 - pw_qpoly_obj)
    )
    assert _is_zero_expression(
        (2 * pw_qpoly_expr).as_isl() - (2 * pw_qpoly_obj)
    )


def _qpolynomial(spec: str) -> isl.QPolynomial:
    return isl.PwQPolynomial(spec).get_pieces()[0][1]


def _assert_expression_equal(
    actual: ScalarIslExpression, expected: ScalarIslExpression
) -> None:
    if isinstance(actual, isl.Aff | isl.PwAff):
        assert actual == expected
        return

    if isinstance(actual, isl.QPolynomial) and isinstance(expected, isl.QPolynomial):
        assert (actual - expected).is_zero()
        return

    if isinstance(actual, isl.PwQPolynomial) and isinstance(
        expected, isl.PwQPolynomial
    ):
        assert (actual - expected).is_zero()
        return

    raise TypeError(
        f"cannot compare {type(actual).__name__} and {type(expected).__name__}"
    )


def as_isl() -> None:
    cases = (
        (
            nisl.make_aff("[n] -> { [i] -> [i + n] }"),
            isl.Aff("[n] -> { [i] -> [i + n] }"),
        ),
        (
            nisl.make_pw_aff("[n] -> { [i] -> [i + n] }"),
            isl.PwAff("[n] -> { [i] -> [i + n] }"),
        ),
        (
            nisl.make_qpolynomial("[n] -> { [i] -> i + n }"),
            _qpolynomial("[n] -> { [i] -> i + n }"),
        ),
        (
            nisl.make_pw_qpolynomial("[n] -> { [i] -> i + n }"),
            isl.PwQPolynomial("[n] -> { [i] -> i + n }"),
        ),
    )

    for named_expr, isl_expr in cases:
        moved = named_expr.move_dims("n", isl.dim_type.in_)
        expected = isl_expr.move_dims(
            isl.dim_type.in_,
            1,
            isl.dim_type.param,
            0,
            1,
        )

        _assert_expression_equal(moved.as_isl(), expected)
        assert moved.input_names == frozenset({"i", "n"})
        assert moved.parameter_names == frozenset()


def as_isl() -> None:
    cases = (
        (
            nisl.make_aff("[n] -> { [i] -> [i + n] }"),
            isl.Aff("[n] -> { [i] -> [i + n] }"),
        ),
        (
            nisl.make_pw_aff("[n] -> { [i] -> [i + n] }"),
            isl.PwAff("[n] -> { [i] -> [i + n] }"),
        ),
        (
            nisl.make_qpolynomial("[n] -> { [i] -> i + n }"),
            _qpolynomial("[n] -> { [i] -> i + n }"),
        ),
        (
            nisl.make_pw_qpolynomial("[n] -> { [i] -> i + n }"),
            isl.PwQPolynomial("[n] -> { [i] -> i + n }"),
        ),
    )

    for named_expr, isl_expr in cases:
        moved = named_expr.move_dims("i", isl.dim_type.param)
        expected = isl_expr.move_dims(
            isl.dim_type.param,
            1,
            isl.dim_type.in_,
            0,
            1,
        )

        _assert_expression_equal(moved.as_isl(), expected)
        assert moved.input_names == frozenset()
        assert moved.parameter_names == frozenset({"n", "i"})

# }}}


def test_multi_aff_get_at_uses_name() -> None:
    map_ = nisl.make_map("{ [i] -> [x = i, y = 2i] }")
    maff = nisl.make_multi_aff(
        map_.as_isl().as_pw_multi_aff().as_multi_aff()
    )
    assert maff.get_at("x").as_isl() == isl.PwAff("{ [i] -> [(i)] }")


def test_pw_multi_aff_get_at_uses_name() -> None:
    map_ = nisl.make_map("{ [i] -> [x = i, y = 2i] }")
    pmaff = nisl.make_pw_multi_aff(map_.as_pw_multi_aff())
    assert pmaff.get_at("y").as_isl() == isl.PwAff("{ [i] -> [(2i)] }")


def test_multi_aff_stores_pw_aff_parts() -> None:
    raw_maff = isl.MultiAff("{ [i] -> [x = i, y = 2i] }")

    maff = nisl.make_multi_aff(raw_maff)

    assert isinstance(maff._obj, constantdict)
    assert not isinstance(maff._obj, isl.Set)
    assert all(isinstance(part, nisl.PwAff) for part in maff._obj.values())
    assert maff.get_at("x") is maff._obj["x"]
    assert maff.get_at("y") is maff._obj["y"]
    assert maff.output_names == frozenset(maff._obj)
    assert maff.input_names == maff.get_at("x").input_names
    assert maff.parameter_names == maff.get_at("x").parameter_names
    assert maff.as_isl() == raw_maff


def test_pw_multi_aff_stores_pw_aff_parts() -> None:
    raw_pmaff = isl.PwMultiAff("[n] -> { [i] -> [x = i + n, y = 2i] }")

    pmaff = nisl.make_pw_multi_aff(raw_pmaff)

    assert isinstance(pmaff._obj, constantdict)
    assert not isinstance(pmaff._obj, isl.Set)
    assert all(isinstance(part, nisl.PwAff) for part in pmaff._obj.values())
    assert pmaff.get_at("x") is pmaff._obj["x"]
    assert pmaff.get_at("y") is pmaff._obj["y"]
    assert pmaff.output_names == frozenset(pmaff._obj)
    assert pmaff.input_names == pmaff.get_at("x").input_names
    assert pmaff.parameter_names == pmaff.get_at("x").parameter_names
    assert pmaff.as_isl() == raw_pmaff


# {{{ qpolynomials

def test_qpolynomial_from_str():
    spec = "[n] -> { [i] -> 2 * i + n }"
    named_qpolynomial = nisl.make_qpolynomial(spec)
    qpolynomial = isl.PwQPolynomial(spec).get_pieces()[0][1]

    print(named_qpolynomial)
    print(qpolynomial)

    assert (named_qpolynomial.as_isl() - qpolynomial).is_zero()


def test_qpolynomial_from_qpolynomial():
    qpolynomial = isl.PwQPolynomial(
        "[n] -> { [i] -> 2 * i + n }").get_pieces()[0][1]
    named_qpolynomial = nisl.make_qpolynomial(qpolynomial)

    print(named_qpolynomial)
    print(qpolynomial)

    assert (named_qpolynomial.as_isl() - qpolynomial).is_zero()


def test_qpolynomial_binary_ops():
    spec = "[n] -> { [i] -> 2 * i + n }"
    named_qpolynomial = nisl.make_qpolynomial(spec)
    qpolynomial = isl.PwQPolynomial(spec).get_pieces()[0][1]

    qpolynomial_p_qpolynomial = qpolynomial + qpolynomial
    qpolynomial_p_1 = qpolynomial + 1

    nqpolynomial_p_nqpolynomial = named_qpolynomial + named_qpolynomial
    nqpolynomial_p_1 = named_qpolynomial + 1

    assert (
        nqpolynomial_p_nqpolynomial.as_isl()
        -
        qpolynomial_p_qpolynomial
    ).is_zero()
    assert (
        nqpolynomial_p_1.as_isl()
        -
        qpolynomial_p_1
    ).is_zero()

    qpolynomial_s_qpolynomial = qpolynomial - qpolynomial
    qpolynomial_s_1 = qpolynomial - 1

    nqpolynomial_s_nqpolynomial = named_qpolynomial - named_qpolynomial
    nqpolynomial_s_1 = named_qpolynomial - 1

    assert (
        nqpolynomial_s_nqpolynomial.as_isl()
        -
        qpolynomial_s_qpolynomial
    ).is_zero()
    assert (
        nqpolynomial_s_1.as_isl()
        -
        qpolynomial_s_1
    ).is_zero()

    qpolynomial_m_3 = qpolynomial * 3
    nqpolynomial_m_3 = named_qpolynomial * 3

    assert (
        nqpolynomial_m_3.as_isl()
        -
        qpolynomial_m_3
    ).is_zero()


def test_qpolynomial_permuted_is_zero():
    # forces the objects to be aligned before binary op is applied, whereas
    # the test above does not necessarily touch alignment code

    named_qp = nisl.make_qpolynomial(
        "[n, m] -> { [a, b] -> a*a + 2*b + n - m }")
    named_qp_perm = nisl.make_qpolynomial(
        "[m, n] -> { [b, a] -> a*a + 2*b + n - m }")

    assert (
        named_qp - named_qp_perm
    ).as_isl().is_zero()

# }}}


# {{{ pwqpolynomials

def test_pw_qp_from_str():
    spec = "[n] -> { [i] -> 2 * i + n }"
    named_pw_qp = nisl.make_pw_qpolynomial(spec)
    pw_qp = isl.PwQPolynomial(spec)

    print(named_pw_qp)
    print(pw_qp)

    assert (named_pw_qp.as_isl() - pw_qp).is_zero()


def test_pw_qp_from_pw_qp():
    pw_qp = isl.PwQPolynomial(
        "[n] -> { [i] -> 2 * i + n }")
    named_pw_qp = nisl.make_pw_qpolynomial(pw_qp)

    print(named_pw_qp)
    print(pw_qp)

    assert (named_pw_qp.as_isl() - pw_qp).is_zero()


def test_pw_qp_binary_ops():
    spec = "[n] -> { [i] -> 2 * i + n }"
    named_pw_qp = nisl.make_pw_qpolynomial(spec)
    pw_qp = isl.PwQPolynomial(spec)

    pw_qp_p_pw_qp = pw_qp + pw_qp
    pw_qp_p_1 = pw_qp + 1

    npw_qp_p_npw_qp = named_pw_qp + named_pw_qp
    npw_qp_p_1 = named_pw_qp + 1

    assert (
        npw_qp_p_npw_qp.as_isl()
        -
        pw_qp_p_pw_qp
    ).is_zero()
    assert (
        npw_qp_p_1.as_isl()
        -
        pw_qp_p_1
    ).is_zero()

    pw_qp_s_pw_qp = pw_qp - pw_qp
    pw_qp_s_1 = pw_qp - 1

    npw_qp_s_npw_qp = named_pw_qp - named_pw_qp
    npw_qp_s_1 = named_pw_qp - 1

    assert (
        npw_qp_s_npw_qp.as_isl()
        -
        pw_qp_s_pw_qp
    ).is_zero()
    assert (
        npw_qp_s_1.as_isl()
        -
        pw_qp_s_1
    ).is_zero()

    pw_qp_m_3 = pw_qp * 3
    npw_qp_m_3 = named_pw_qp * 3

    assert (
        npw_qp_m_3.as_isl()
        -
        pw_qp_m_3
    ).is_zero()


def test_pw_qpolynomial_permuted_is_zero():
    # forces the objects to be aligned before binary op is applied, whereas
    # the test above does not necessarily touch alignment code

    named_qp = nisl.make_pw_qpolynomial(
        "[n, m] -> { [a, b] -> a*a + 2*b + n - m }")
    named_qp_perm = nisl.make_pw_qpolynomial(
        "[m, n] -> { [b, a] -> a*a + 2*b + n - m }")

    assert (
        named_qp - named_qp_perm
    ).is_zero()

# }}}


# {{{ multiaffs

# }}}


# {{{ pwmultiaffs

# }}}
