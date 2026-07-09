"""
Name-aware affine and polynomial expression wrappers.

The wrappers in this module provide a small arithmetic interface around isl
expression objects while preserving named dimension metadata across alignment
and reconstruction.
"""

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
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    TypeVar,
    cast,
    final,
    overload,
)

from typing_extensions import Self, override

import islpy as isl

from .core import (
    DimType,
    IslExpressionLikeT_co,
    IslMultiExpressionLikeT_co,
    IslPolynomialLikeT_co,
    IslScalarExpressionLike,
    IslScalarExpressionLikeT,
    NamedIslObject,
    Space,
    align_two,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from namedisl.set_like import Set


NamedExpressionLikeT_co = TypeVar(
    "NamedExpressionLikeT_co",
    bound=IslScalarExpressionLike,
    covariant=True,
)


def _apply_expression_binary_op(
    lhs: _NamedExpressionLike[IslScalarExpressionLikeT] | int,
    rhs: _NamedExpressionLike[IslScalarExpressionLikeT] | int,
    op: Callable[
        [IslScalarExpressionLikeT | int, IslScalarExpressionLikeT | int],
        IslScalarExpressionLikeT | int,
    ],
) -> _NamedExpressionLike[IslScalarExpressionLikeT]:
    if isinstance(rhs, int):
        if isinstance(lhs, int):
            raise TypeError("both types are int")

        return type(lhs)(
            cast("IslScalarExpressionLikeT", op(lhs._obj, rhs)),
            lhs.space,
        )
    if isinstance(lhs, int):
        return type(rhs)(
            cast("IslScalarExpressionLikeT", op(lhs, rhs._obj)),
            rhs.space,
        )

    if type(lhs) is not type(rhs):
        return NotImplemented

    lhs, rhs = align_two(lhs, rhs)
    return type(lhs)(
        cast("IslScalarExpressionLikeT", op(lhs._obj, rhs._obj)), lhs.space)


# {{{ "base" named expression-likes (affs, pwaffs, qpolynomials, pwqpolynomials)

@dataclass(frozen=True, eq=False)
class _NamedExpressionLike(NamedIslObject[IslExpressionLikeT_co]):
    """
    .. autoattribute:: active_dim_types
    .. automethod:: __add__
    .. automethod:: __radd__
    .. automethod:: __sub__
    .. automethod:: __rsub__
    .. automethod:: __mul__
    .. automethod:: __rmul__
    .. automethod:: __bool__
    .. automethod:: is_zero
    .. automethod:: __eq__
    .. automethod:: equals
    """
    active_dim_types: ClassVar[frozenset[DimType]] = frozenset({
        DimType.param, DimType.in_})

    def __add__(
        self: Self,
        other: Self | int,
    ) -> Self:
        return cast("Self", _apply_expression_binary_op(self, other, operator.add))

    def __radd__(
        self: Self,
        other: int,
    ) -> Self:
        return cast("Self", _apply_expression_binary_op(other, self, operator.add))

    def __sub__(
        self: Self,
        other: Self | int,
    ) -> Self:
        return cast("Self", _apply_expression_binary_op(self, other, operator.sub))

    def __rsub__(
        self: Self,
        other: int,
    ) -> Self:
        return cast("Self", _apply_expression_binary_op(other, self, operator.sub))

    def __mul__(
        self: Self,
        other: Self | int,
    ) -> Self:
        return cast("Self", _apply_expression_binary_op(self, other, operator.mul))

    def __rmul__(
        self: Self,
        other: int,
    ) -> Self:
        return cast("Self", _apply_expression_binary_op(other, self, operator.mul))

    def __bool__(self) -> bool:
        if isinstance(self._obj,
                      (isl.PwAff, isl.PwQPolynomial, isl.QPolynomial,
                          isl.MultiAff, isl.PwMultiAff)):
            raise NotImplementedError
        return bool(self._obj.plain_is_zero())

    def is_zero(self) -> bool:
        if isinstance(self._obj,
                      (isl.PwMultiAff, isl.MultiAff, isl.PwAff, isl.Aff)):
            raise NotImplementedError
        return bool(self._obj.is_zero())

    @override
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        other = cast("Self", other)

        if not self.space.order_equal(other.space):
            return False

        if isinstance(self._obj, (isl.QPolynomial, isl.PwQPolynomial)):
            return (self._obj - other._obj).is_zero()
        elif isinstance(self._obj, isl.MultiAff):
            raise NotImplementedError()
        elif isinstance(self._obj, isl.PwMultiAff):
            return self._obj.is_equal(other._obj)
        else:
            return self._obj.plain_is_equal(other._obj)

    def equals(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        other = cast("Self", other)

        aligned_lhs, aligned_rhs = align_two(self, other)

        if isinstance(aligned_lhs._obj, (isl.QPolynomial, isl.PwQPolynomial)):
            return (aligned_lhs._obj - aligned_rhs._obj).is_zero()
        elif isinstance(aligned_lhs._obj, (isl.Aff, isl.MultiAff)):
            # FIXME: I *think* plain should be exact for Aff? It does not have is_equal.
            return aligned_lhs._obj.plain_is_equal(aligned_rhs._obj)
        else:
            return aligned_lhs._obj.is_equal(aligned_rhs._obj)


@dataclass(frozen=True, eq=False)
class Aff(_NamedExpressionLike[isl.Aff]):
    """
    .. automethod:: zero_on_domain
    .. automethod:: from_space
    .. automethod:: set_coefficient
    .. automethod:: as_pw_aff
    """
    @staticmethod
    def zero_on_domain(space: Space) -> Aff:
        return Aff(
            isl.Aff.zero_on_domain(isl.LocalSpace.from_space(space.as_isl_set_space())),
            space.as_expr_space())

    @staticmethod
    def from_space(space: Space) -> dict[str | Literal[0], Aff]:
        """*space* is assumed to be a set-like space."""
        zero = Aff.zero_on_domain(space)
        result: dict[str | Literal[0], Aff] = {0: zero}

        expr_space = zero.space
        isl_zero = zero._obj
        for name, (dt, idx) in expr_space.name_to_dim.items():
            result[name] = Aff(
                isl_zero.set_coefficient_val(dt.as_isl(), idx, 1),
                expr_space)

        return result

    def set_coefficient(self, name: str, value: int) -> Aff:
        dt, idx = self.space.name_to_dim[name]
        return Aff(self._obj.set_coefficient_val(dt.as_isl(), idx, value), self.space)

    def as_pw_aff(self) -> PwAff:
        return PwAff(self._obj.to_pw_aff(), self.space)


@overload
def make_aff(src: str, ctx: isl.Context | None = None) -> Aff:
    ...


@overload
def make_aff(src: isl.Aff) -> Aff:
    ...


def make_aff(src: str | isl.Aff, ctx: isl.Context | None = None) -> Aff:
    """
    Create an :class:`Aff` from isl syntax or an :class:`islpy.Aff`.
    """
    obj = isl.Aff(src, ctx) if isinstance(src, str) else src
    return Aff(obj, Space.from_isl(obj, Aff.active_dim_types))


@dataclass(frozen=True, eq=False)
class PwAff(_NamedExpressionLike[isl.PwAff]):
    """
    .. automethod:: gist
    .. automethod:: from_space
    .. automethod:: where
    """
    def gist(self, set: Set):
        self_a, set_a = align_two(self, set)
        result = self_a._obj.gist(set_a._obj)
        return PwAff(result, self_a.space)

    @staticmethod
    def from_space(space: Space) -> dict[str | Literal[0], PwAff]:
        """*space* is assumed to be a set-like space."""
        zero = Aff.zero_on_domain(space)
        result: dict[str | Literal[0], PwAff] = {0: zero.as_pw_aff()}

        expr_space = zero.space
        isl_zero = zero._obj
        for name, (dt, idx) in expr_space.name_to_dim.items():
            result[name] = PwAff(
                isl_zero.set_coefficient_val(dt.as_isl(), idx, 1).to_pw_aff(),
                expr_space)

        return result

    _op_to_func: ClassVar[dict[str, Callable[[isl.PwAff, isl.PwAff], isl.Set]]] = {
        "<": isl.PwAff.lt_set,
        "<=": isl.PwAff.le_set,
        "=": isl.PwAff.eq_set,
        "!=": isl.PwAff.ne_set,
        ">": isl.PwAff.gt_set,
        ">=": isl.PwAff.ge_set,
    }

    def where(self,
        op: Literal["<", "<=", "=", "!=", ">=", ">"],
        rhs: int | Aff | PwAff
    ) -> Set:
        func = self._op_to_func[op]
        if isinstance(rhs, int):
            rhs = PwAff(isl.PwAff.zero_on_domain(self._obj.space) + rhs, self.space)
        elif isinstance(rhs, Aff):
            rhs = rhs.as_pw_aff()
        self_a, rhs_a = align_two(self, rhs)
        from .set_like import Set
        return Set(func(self_a._obj, rhs_a._obj), self_a.space.as_set_space())


@overload
def make_pw_aff(src: str, ctx: isl.Context | None = None) -> PwAff:
    ...


@overload
def make_pw_aff(src: isl.PwAff) -> PwAff:
    ...


def make_pw_aff(src: str | isl.PwAff, ctx: isl.Context | None = None) -> PwAff:
    """
    Create a :class:`PwAff` from isl syntax or an :class:`islpy.PwAff`.
    """
    obj = isl.PwAff(src, ctx) if isinstance(src, str) else src
    return PwAff(obj, Space.from_isl(obj, PwAff. active_dim_types))


@dataclass(frozen=True, eq=False)
class _NamedPolynomialLike(_NamedExpressionLike[IslPolynomialLikeT_co]):
    """
    """
    pass


@dataclass(frozen=True, eq=False)
class QPolynomial(_NamedPolynomialLike[isl.QPolynomial]):
    """
    """
    pass


@overload
def make_qpolynomial(src: str, ctx: isl.Context | None = None) -> QPolynomial:
    ...


@overload
def make_qpolynomial(src: isl.QPolynomial) -> QPolynomial:
    ...


def make_qpolynomial(
        src: str | isl.QPolynomial, ctx: isl.Context | None = None) -> QPolynomial:
    """
    Create a :class:`QPolynomial` from isl syntax or an isl qpolynomial.
    """
    # NOTE: ISL does not have a QPolynomial constructor, but we can make one
    # here by first creating a PwQPolynomial, then taking the only QPolynomial
    # that comes out of it :shrug:
    if isinstance(src, str):
        (_where, obj), = isl.PwQPolynomial(src, ctx).get_pieces()
    else:
        obj = src

    return QPolynomial(obj, Space.from_isl(obj, QPolynomial.active_dim_types))


@dataclass(frozen=True, eq=False)
class PwQPolynomial(_NamedPolynomialLike[isl.PwQPolynomial]):
    """
    """
    pass


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
    """
    Create a :class:`PwQPolynomial` from isl syntax or an isl object.
    """
    obj = isl.PwQPolynomial(src, ctx) if isinstance(src, str) else src
    return PwQPolynomial(
        obj,
        Space.from_isl(obj, PwQPolynomial.active_dim_types))


# }}}


# {{{ multi expression-likes (multiaff, pwmultiaff)

@dataclass(frozen=True, eq=False)
class _NamedMultiExpressionLike(_NamedExpressionLike[IslMultiExpressionLikeT_co]):
    """
    .. autoattribute:: active_dim_types
    """
    active_dim_types: ClassVar[frozenset[DimType]] = frozenset({
        DimType.param, DimType.in_, DimType.out})


@final
@dataclass(frozen=True, eq=False)
class MultiAff(_NamedMultiExpressionLike[isl.MultiAff]):
    """
    .. automethod:: __getitem__
    """
    def __getitem__(self, name: str):
        dt, idx = self.space.name_to_dim[name]
        if dt != DimType.out:
            raise ValueError(f"'{name}' does not name an output dimension")
        return Aff(self._obj.get_at(idx), self.space.drop_dim_type(DimType.out))


@overload
def make_multi_aff(src: str, ctx: isl.Context | None = None) -> MultiAff:
    ...


@overload
def make_multi_aff(src: isl.MultiAff) -> MultiAff:
    ...


def make_multi_aff(
        src: str | isl.MultiAff, ctx: isl.Context | None = None) -> MultiAff:
    """
    Create a :class:`MultiAff` from isl syntax or an :class:`islpy.MultiAff`.
    """
    obj = isl.MultiAff(src, ctx) if isinstance(src, str) else src
    return MultiAff(obj, Space.from_isl(obj, MultiAff.active_dim_types))


@dataclass(frozen=True, eq=False)
class PwMultiAff(_NamedMultiExpressionLike[isl.PwMultiAff]):
    """
    .. automethod:: __getitem__
    .. automethod:: as_multi_aff
    """
    def __getitem__(self, name: str):
        dt, idx = self.space.name_to_dim[name]
        if dt != DimType.out:
            raise ValueError(f"'{name}' does not name an output dimension")
        return PwAff(self._obj.get_at(idx), self.space.drop_dim_type(DimType.out))

    def as_multi_aff(self):
        return MultiAff(self._obj.as_multi_aff(), self.space)


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
    return PwMultiAff(obj, Space.from_isl(obj, PwMultiAff.active_dim_types))

# }}}
