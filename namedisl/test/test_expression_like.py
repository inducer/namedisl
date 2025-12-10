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

# {{{ multi-expression objects

def test_pw_multi_aff_from_str() -> None:
    pw_multi_aff = nisl.make_pw_multi_aff(
        "{ [i, j] -> [2 * i, 2 * j] : 0 <= i, j < 20 }"
    )

    print(pw_multi_aff._obj)
    print(pw_multi_aff)


def test_pw_multi_aff_from_pw_multi_aff() -> None:
    m = isl.Map("""{
        [i, j] -> [io, ii, jo, ji] :
        0 <= ii, ji < 32 and
        i = io * 32 + ii and
        j = jo * 32 + ji
    }""")

    pw_multi_aff = nisl.make_pw_multi_aff(m.as_pw_multi_aff())
    print(pw_multi_aff._obj)
    print(pw_multi_aff)


def test_pw_multi_aff_get_at() -> None:
    m = nisl.make_map("""{
        [i, j] -> [io, ii, jo, ji] :
        0 <= ii, ji < 32 and
        i = io * 32 + ii and
        j = jo * 32 + ji
    }""")

    pw_multi_aff = m.as_pw_multi_aff()

    at = pw_multi_aff.get_at("i")

    print(at._obj)
    print(at)


# }}}


# {{{ piece-wise expression objects

def test_pw_aff_from_str() -> None:
    pw_aff = nisl.make_pw_aff("{ [i] -> [2 * i] : 0 <= i < 10 }")
    print(pw_aff._obj)
    print(pw_aff)


def test_pw_aff_from_pw_aff() -> None:
    pw_aff = isl.PwAff("{ [i] -> [2 * i] : 0 <= i < 10 }")
    named_pw_aff = nisl.make_pw_aff(pw_aff)

    print(named_pw_aff._obj)
    print(named_pw_aff)


def test_pw_aff_get_pieces() -> None:
    m = nisl.make_map("{ [i] -> [io, ii] : 0 <= ii < 32 and i = io * 32 + ii }")
    pw_multi_aff = m.as_pw_multi_aff()
    pw_aff = pw_multi_aff.get_at("i")

    for (dom, expn) in pw_aff.get_pieces():  # type: ignore
        print(f"Domain = {dom._obj}")
        print(f"Named domain = {dom}")

        print(f"Expression = {expn._obj}")
        print(f"Named expression = {expn}")


def test_pw_qpolynomial_from_str() -> None:
    pw_qpoly = nisl.make_pw_qpolynomial("{ [i] -> i^2 + 2*i + 3 }")
    print(pw_qpoly._obj)
    print(pw_qpoly)


def test_pw_qpolynomial_from_pw_qpolynomial() -> None:
    pw_qpoly = isl.PwQPolynomial("{ [i] -> i^2 + 2*i + 3 }")
    named_pw_qpoly = nisl.make_pw_qpolynomial(pw_qpoly)

    print(named_pw_qpoly._obj)
    print(named_pw_qpoly)


def test_pw_qpolynomial_get_pieces() -> None:
    pw_qpoly = nisl.make_pw_qpolynomial("{ [i] -> i^2 + 2*i + 3 }")
    pieces = pw_qpoly.get_pieces()

    for (dom, expn) in pieces:
        print(f"Domain = {dom._obj}")
        print(f"Named domain = {dom}")

        print(f"Expression = {expn._obj}")
        print(f"Named expression = {expn}")

# }}}
