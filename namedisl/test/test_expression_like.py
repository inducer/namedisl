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

import islpy as isl

import namedisl as nisl


# {{{ affs

def test_aff_from_str():
    spec = "[n] -> { [i] -> [2 * i + n] }"
    named_aff = nisl.make_aff(spec)
    aff = isl.Aff(spec)

    print(named_aff)
    print(aff)

    assert aff == named_aff._reconstruct_isl_object()


def test_aff_from_aff():
    aff = isl.Aff("[n] -> { [i] -> [2 * i + n] }")
    named_aff = nisl.make_aff(aff)

    print(named_aff)
    print(aff)

    assert aff == named_aff._reconstruct_isl_object()


def test_aff_binary_ops():
    spec = "[n] -> { [i] -> [2 * i + n] }"
    named_aff = nisl.make_aff(spec)
    aff = isl.Aff(spec)

    aff_p_aff = aff + aff
    aff_p_1 = aff + 1

    naff_p_naff = named_aff + named_aff
    naff_p_1 = named_aff + 1

    assert naff_p_naff._reconstruct_isl_object() == aff_p_aff
    assert naff_p_1._reconstruct_isl_object() == aff_p_1

    aff_s_aff = aff - aff
    aff_s_1 = aff - 1

    naff_s_naff = named_aff - named_aff
    naff_s_1 = named_aff - 1

    assert naff_s_naff._reconstruct_isl_object() == aff_s_aff
    assert naff_s_1._reconstruct_isl_object() == aff_s_1

    aff_m_3 = aff * 3
    naff_m_3 = named_aff * 3

    assert naff_m_3._reconstruct_isl_object() == aff_m_3

# }}}


# {{{ pwaffs

def test_pwaff_from_str():
    spec = "[n] -> { [i] -> [2 * i + n] }"
    named_pwaff = nisl.make_pw_aff(spec)
    pwaff = isl.PwAff(spec)

    print(named_pwaff)
    print(pwaff)

    assert pwaff == named_pwaff._reconstruct_isl_object()


def test_pwaff_from_pwaff():
    pwaff = isl.PwAff("[n] -> { [i] -> [2 * i + n] }")
    named_pwaff = nisl.make_pw_aff(pwaff)

    print(named_pwaff)
    print(pwaff)

    assert pwaff == named_pwaff._reconstruct_isl_object()


def test_pwaff_binary_ops():
    spec = "[n] -> { [i] -> [2 * i + n] }"
    named_pwaff = nisl.make_pw_aff(spec)
    pwaff = isl.PwAff(spec)

    pwaff_p_pwaff = pwaff + pwaff
    pwaff_p_1 = pwaff + 1

    npwaff_p_npwaff = named_pwaff + named_pwaff
    npwaff_p_1 = named_pwaff + 1

    assert npwaff_p_npwaff._reconstruct_isl_object() == pwaff_p_pwaff
    assert npwaff_p_1._reconstruct_isl_object() == pwaff_p_1

    pwaff_s_pwaff = pwaff - pwaff
    pwaff_s_1 = pwaff - 1

    npwaff_s_npwaff = named_pwaff - named_pwaff
    npwaff_s_1 = named_pwaff - 1

    assert npwaff_s_npwaff._reconstruct_isl_object() == pwaff_s_pwaff
    assert npwaff_s_1._reconstruct_isl_object() == pwaff_s_1

    pwaff_m_3 = pwaff * 3
    npwaff_m_3 = named_pwaff * 3

    assert npwaff_m_3._reconstruct_isl_object() == pwaff_m_3


def test_mixed_aff_and_pwaff_binary_op_promotes_to_pwaff() -> None:
    aff = nisl.make_aff("{ [i] -> [i] }")
    pwaff = nisl.make_pw_aff("{ [i] -> [i] }")

    result = aff + pwaff

    assert isinstance(result, nisl.PwAff)
    assert result._reconstruct_isl_object() == (
        aff._reconstruct_isl_object().to_pw_aff() + pwaff._reconstruct_isl_object()
    )

# }}}


def test_multi_aff_get_at_uses_name() -> None:
    map_ = nisl.make_map("{ [i] -> [x = i, y = 2i] }")
    maff = nisl.make_multi_aff(
        map_._reconstruct_isl_object().as_pw_multi_aff().as_multi_aff()
    )
    assert maff.get_at("x")._reconstruct_isl_object() == isl.Aff("{ [i] -> [(i)] }")


def test_pw_multi_aff_get_at_uses_name() -> None:
    map_ = nisl.make_map("{ [i] -> [x = i, y = 2i] }")
    pmaff = nisl.make_pw_multi_aff(map_.as_pw_multi_aff())
    assert pmaff.get_at("y")._reconstruct_isl_object() == isl.PwAff("{ [i] -> [(2i)] }")


# {{{ qpolynomials

def test_qpolynomial_from_str():
    spec = "[n] -> { [i] -> 2 * i + n }"
    named_qpolynomial = nisl.make_qpolynomial(spec)
    qpolynomial = isl.PwQPolynomial(spec).get_pieces()[0][1]

    print(named_qpolynomial)
    print(qpolynomial)

    assert (named_qpolynomial._reconstruct_isl_object() - qpolynomial).is_zero()


def test_qpolynomial_from_qpolynomial():
    qpolynomial = isl.PwQPolynomial(
        "[n] -> { [i] -> 2 * i + n }").get_pieces()[0][1]
    named_qpolynomial = nisl.make_qpolynomial(qpolynomial)

    print(named_qpolynomial)
    print(qpolynomial)

    assert (named_qpolynomial._reconstruct_isl_object() - qpolynomial).is_zero()


def test_qpolynomial_binary_ops():
    spec = "[n] -> { [i] -> 2 * i + n }"
    named_qpolynomial = nisl.make_qpolynomial(spec)
    qpolynomial = isl.PwQPolynomial(spec).get_pieces()[0][1]

    qpolynomial_p_qpolynomial = qpolynomial + qpolynomial
    qpolynomial_p_1 = qpolynomial + 1

    nqpolynomial_p_nqpolynomial = named_qpolynomial + named_qpolynomial
    nqpolynomial_p_1 = named_qpolynomial + 1

    assert (
        nqpolynomial_p_nqpolynomial._reconstruct_isl_object()
        -
        qpolynomial_p_qpolynomial
    ).is_zero()
    assert (
        nqpolynomial_p_1._reconstruct_isl_object()
        -
        qpolynomial_p_1
    ).is_zero()

    qpolynomial_s_qpolynomial = qpolynomial - qpolynomial
    qpolynomial_s_1 = qpolynomial - 1

    nqpolynomial_s_nqpolynomial = named_qpolynomial - named_qpolynomial
    nqpolynomial_s_1 = named_qpolynomial - 1

    assert (
        nqpolynomial_s_nqpolynomial._reconstruct_isl_object()
        -
        qpolynomial_s_qpolynomial
    ).is_zero()
    assert (
        nqpolynomial_s_1._reconstruct_isl_object()
        -
        qpolynomial_s_1
    ).is_zero()

    qpolynomial_m_3 = qpolynomial * 3
    nqpolynomial_m_3 = named_qpolynomial * 3

    assert (
        nqpolynomial_m_3._reconstruct_isl_object()
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
    )._reconstruct_isl_object().is_zero()

# }}}


# {{{ pwqpolynomials

def test_pw_qp_from_str():
    spec = "[n] -> { [i] -> 2 * i + n }"
    named_pw_qp = nisl.make_pw_qpolynomial(spec)
    pw_qp = isl.PwQPolynomial(spec)

    print(named_pw_qp)
    print(pw_qp)

    assert (named_pw_qp._reconstruct_isl_object() - pw_qp).is_zero()


def test_pw_qp_from_pw_qp():
    pw_qp = isl.PwQPolynomial(
        "[n] -> { [i] -> 2 * i + n }")
    named_pw_qp = nisl.make_pw_qpolynomial(pw_qp)

    print(named_pw_qp)
    print(pw_qp)

    assert (named_pw_qp._reconstruct_isl_object() - pw_qp).is_zero()


def test_pw_qp_binary_ops():
    spec = "[n] -> { [i] -> 2 * i + n }"
    named_pw_qp = nisl.make_pw_qpolynomial(spec)
    pw_qp = isl.PwQPolynomial(spec)

    pw_qp_p_pw_qp = pw_qp + pw_qp
    pw_qp_p_1 = pw_qp + 1

    npw_qp_p_npw_qp = named_pw_qp + named_pw_qp
    npw_qp_p_1 = named_pw_qp + 1

    assert (
        npw_qp_p_npw_qp._reconstruct_isl_object()
        -
        pw_qp_p_pw_qp
    ).is_zero()
    assert (
        npw_qp_p_1._reconstruct_isl_object()
        -
        pw_qp_p_1
    ).is_zero()

    pw_qp_s_pw_qp = pw_qp - pw_qp
    pw_qp_s_1 = pw_qp - 1

    npw_qp_s_npw_qp = named_pw_qp - named_pw_qp
    npw_qp_s_1 = named_pw_qp - 1

    assert (
        npw_qp_s_npw_qp._reconstruct_isl_object()
        -
        pw_qp_s_pw_qp
    ).is_zero()
    assert (
        npw_qp_s_1._reconstruct_isl_object()
        -
        pw_qp_s_1
    ).is_zero()

    pw_qp_m_3 = pw_qp * 3
    npw_qp_m_3 = named_pw_qp * 3

    assert (
        npw_qp_m_3._reconstruct_isl_object()
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
