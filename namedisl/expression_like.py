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

import operator
from dataclasses import dataclass, replace
from typing import final, overload

from typing_extensions import Self, override

import islpy as isl

from .core import (
    IslExpressionLikeT,
    NamedIslObject,
    _align_and_apply_binary_op,
    _deconstruct_object,
    _strip_names,
)


# {{{ "base" named expression-likes (affs, pwaffs, qpolynomials, pwqpolynomials)

@dataclass(frozen=True, eq=False)
class _NamedExpressionLike(NamedIslObject[IslExpressionLikeT]):
    # FIXME: Self is used here is because _NamedExpressionLike is generic,
    # leading to complaints from basedpyright
    def __add__(self, other: Self | int) -> Self:
        if isinstance(other, int):
            return replace(
                self,
                _obj=operator.add(self._obj, other),
                _name_to_dim=self._name_to_dim,
                _dimtype_to_names=self._dimtype_to_names
            )

        return _align_and_apply_binary_op(self, other, operator.add)

    def __sub__(self, other: Self | int) -> Self:
        if isinstance(other, int):
            return replace(
                self,
                _obj=operator.sub(self._obj, other),
                _name_to_dim=self._name_to_dim,
                _dimtype_to_names=self._dimtype_to_names
            )

        return _align_and_apply_binary_op(self, other, operator.sub)

    def __mul__(self, other: Self | int) -> Self:
        if isinstance(other, int):
            return replace(
                self,
                _obj=operator.mul(self._obj, other),
                _name_to_dim=self._name_to_dim,
                _dimtype_to_names=self._dimtype_to_names
            )

        return _align_and_apply_binary_op(self, other, operator.mul)

    def is_zero(self) -> bool:
        return self._reconstruct_isl_object().is_zero()

    @override
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class _NamedPwExpressionLike(_NamedExpressionLike[IslExpressionLikeT]):
    ...


@final
@dataclass(frozen=True, eq=False)
class Aff(_NamedExpressionLike[isl.Aff]):
    _obj: isl.Aff


@overload
def make_aff(src: str, ctx: isl.Context | None = None) -> Aff:
    ...


@overload
def make_aff(src: isl.Aff) -> Aff:
    ...


def make_aff(src: str | isl.Aff, ctx: isl.Context | None = None) -> Aff:
    obj = isl.Aff(src, ctx) if isinstance(src, str) else src

    aff_obj, dimtype_to_names = _deconstruct_object(obj)
    aff_obj, name_to_dim = _strip_names(aff_obj)

    return Aff(aff_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class QPolynomial(_NamedExpressionLike[isl.QPolynomial]):
    _obj: isl.QPolynomial


@overload
def make_qpolynomial(src: str, ctx: isl.Context | None = None) -> QPolynomial:
    ...


@overload
def make_qpolynomial(src: isl.QPolynomial) -> QPolynomial:
    ...


def make_qpolynomial(
        src: str | isl.QPolynomial, ctx: isl.Context | None = None) -> QPolynomial:
    # NOTE: ISL does not have a QPolynomial constructor, but we can make one
    # here by first creating a PwQPolynomial, then taking the only QPolynomial
    # that comes out of it :shrug:
    obj = (
        isl.PwQPolynomial(src, ctx).get_pieces()[0][1] if isinstance(src, str)
        else src
    )

    qp_obj, dimtype_to_names = _deconstruct_object(obj)
    qp_obj, name_to_dim = _strip_names(qp_obj)

    return QPolynomial(qp_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class PwAff(_NamedPwExpressionLike):
    _obj: isl.PwAff


@overload
def make_pw_aff(src: str, ctx: isl.Context | None = None) -> PwAff:
    ...


@overload
def make_pw_aff(src: isl.PwAff) -> PwAff:
    ...


def make_pw_aff(src: str | isl.PwAff, ctx: isl.Context | None = None) -> PwAff:
    obj = isl.PwAff(src, ctx) if isinstance(src, str) else src

    pwaff_obj, dimtype_to_names = _deconstruct_object(obj)
    pwaff_obj, name_to_dim = _strip_names(pwaff_obj)

    return PwAff(pwaff_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class PwQPolynomial(_NamedPwExpressionLike):
    _obj: isl.PwQPolynomial


@overload
def make_pw_qpolynomial(
        src: str, ctx: isl.Context | None = None) -> PwQPolynomial:
    ...


@overload
def make_pw_qpolynomial(src: isl.PwQPolynomial) -> PwQPolynomial:
    ...


def make_pw_qpolynomial(
        src: str | isl.PwQPolynomial,
        ctx: isl.Context | None = None
    ) -> PwQPolynomial:
    obj = isl.PwQPolynomial(src, ctx) if isinstance(src, str) else src

    pw_qp_obj, dimtype_to_names = _deconstruct_object(obj)
    pw_qp_obj, name_to_dim = _strip_names(pw_qp_obj)

    return PwQPolynomial(pw_qp_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args

# }}}


# {{{ multi expression-likes (multiaff, pwmultiaff)

@dataclass(frozen=True, eq=False)
class _NamedMultiExpressionLike(NamedIslObject[isl.Set]):
    """
    Multi-expressions in ISL cannot have dimensions moved. As a workaround, we
    represent multi-expressions as sets internally. This is done during
    deconstruction by converting a multi-expression to a map, then converting
    the resulting map to a set. During reconstruction, we simply follow the
    deconstruction steps backwards (set -> map -> multi-expression). As such,
    reconstruction is special-cased for each subclass.
    """


@final
@dataclass(frozen=True, eq=False)
class PwMultiAff(_NamedMultiExpressionLike):
    @override
    def _reconstruct_isl_object(self) -> isl.PwMultiAff:
        # deconstruction: isl.PwMultiAff -> isl.Map -> isl.Set
        # reconstruction: isl.Set -> isl.Map -> isl.PwMultiAff
        return super()._reconstruct_isl_object().as_pw_multi_aff()


@overload
def make_pw_multi_aff(src: str, ctx: isl.Context | None = None) -> PwMultiAff:
    ...


@overload
def make_pw_multi_aff(src: isl.PwMultiAff) -> PwMultiAff:
    ...


def make_pw_multi_aff(
        src: str | isl.PwMultiAff,
        ctx: isl.Context | None = None
    ) -> PwMultiAff:

    obj = isl.PwMultiAff(src, ctx) if isinstance(src, str) else src

    pw_maff_obj, dimtype_to_names = _deconstruct_object(obj)
    pw_maff_obj, name_to_dim = _strip_names(pw_maff_obj)

    return PwMultiAff(pw_maff_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class MultiAff(NamedIslObject[isl.Set]):
    @override
    def _reconstruct_isl_object(self) -> isl.MultiAff:
        # deconstruction: isl.MultiAff -> isl.Map -> isl.Set
        # reconstruction: isl.Set -> isl.Map -> isl.PwMultiAff -> isl.MultiAff
        return super()._reconstruct_isl_object().as_pw_multi_aff().as_multi_aff()


@overload
def make_multi_aff(src: str, ctx: isl.Context | None = None) -> MultiAff:
    ...


@overload
def make_multi_aff(src: isl.MultiAff) -> MultiAff:
    ...


def make_multi_aff(
        src: str | isl.MultiAff, ctx: isl.Context | None = None) -> MultiAff:
    obj = isl.MultiAff(src, ctx) if isinstance(src, str) else src

    maff_obj, dimtype_to_names = _deconstruct_object(obj)
    maff_obj, name_to_dim = _strip_names(maff_obj)

    return MultiAff(maff_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args

# }}}
