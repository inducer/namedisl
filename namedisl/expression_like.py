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
from dataclasses import dataclass
from typing import final, overload

from typing_extensions import Self, override

import islpy as isl

from .core import (
    IslExpressionLike,
    NamedIslObject,
    _align_and_apply_binary_op,
    _deconstruct_object,
    _strip_names,
)


@dataclass(frozen=True, eq=False)
class _NamedExpressionLike(NamedIslObject[IslExpressionLike]):
    @override
    def _reconstruct_isl_object(self) -> IslExpressionLike:
        return self._obj

    # FIXME: Self is used here is because _NamedExpressionLike is generic,
    # leading to complaints from basedpyright
    def __add__(self, other: Self) -> Self:
        return _align_and_apply_binary_op(self, other, operator.add)

    def __sub__(self, other: Self) -> Self:
        return _align_and_apply_binary_op(self, other, operator.sub)

    def __mul__(self, other: Self) -> Self:
        return _align_and_apply_binary_op(self, other, operator.mul)

    @override
    def __eq__(self, other: object) -> bool:
        assert type(other) is type(self)
        return self._obj == other._obj


@dataclass(frozen=True, eq=False)
class _NamedPwExpressionLike(_NamedExpressionLike):
    ...


@dataclass(frozen=True, eq=False)
class _NamedMultiExpressionLike(_NamedExpressionLike):
    ...


@final
@dataclass(frozen=True, eq=False)
class Aff(_NamedExpressionLike):
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

    assert isinstance(aff_obj, isl.Aff)
    aff_obj, name_to_dim = _strip_names(aff_obj)

    return Aff(aff_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class PwAff(_NamedPwExpressionLike):
    _obj: isl.PwAff


@overload
def make_pwaff(src: str, ctx: isl.Context | None = None) -> PwAff:
    ...


@overload
def make_pwaff(src: isl.PwAff) -> PwAff:
    ...


def make_pwaff(src: str | isl.PwAff, ctx: isl.Context | None = None) -> PwAff:
    obj = isl.PwAff(src, ctx) if isinstance(src, str) else src

    pwaff_obj, dimtype_to_names = _deconstruct_object(obj)

    assert isinstance(obj, isl.PwAff)
    pwaff_obj, name_to_dim = _strip_names(obj)

    return PwAff(pwaff_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class PwMultiAff(_NamedMultiExpressionLike):
    _obj: isl.PwMultiAff


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

    assert isinstance(pw_maff_obj, isl.PwMultiAff)
    pw_maff_obj, name_to_dim = _strip_names(pw_maff_obj)

    return PwMultiAff(pw_maff_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args
