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
from typing import TYPE_CHECKING, ClassVar, Literal, TypeVar, cast, final, overload

from constantdict import constantdict
from typing_extensions import Self, override

import islpy as isl

from .core import (
    DimType,
    DimTypeToNames,
    IslExpressionLike,
    IslExpressionLikeT,
    IslMultiExpressionLikeT_co,
    NamedIslObject,
    NameToDim,
    Space,
    _align_two,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from namedisl.set_like import Set


NamedExpressionLikeT_co = TypeVar(
    "NamedExpressionLikeT_co",
    bound=IslExpressionLike,
    covariant=True,
)


def _apply_expression_binary_op(
    lhs: _NamedExpressionLike[IslExpressionLikeT] | int,
    rhs: _NamedExpressionLike[IslExpressionLikeT] | int,
    op: Callable[
        [IslExpressionLikeT | int, IslExpressionLikeT | int],
        IslExpressionLikeT | int,
    ],
) -> _NamedExpressionLike[IslExpressionLikeT]:
    if isinstance(rhs, int):
        if isinstance(lhs, int):
            raise TypeError("both types are int")

        return type(lhs)(
            op(lhs._obj, rhs),
            lhs.sp,
        )
    if isinstance(lhs, int):
        return type(rhs)(
            op(lhs, rhs._obj),
            rhs.sp,
        )

    lhs, rhs = _align_two(lhs, rhs)
    return type(lhs)(cast("IslExpressionLikeT", op(lhs._obj, rhs._obj)), lhs.sp)


# {{{ "base" named expression-likes (affs, pwaffs, qpolynomials, pwqpolynomials)

@dataclass(frozen=True, eq=False)
class _NamedExpressionLike(
    NamedIslObject[IslExpressionLikeT]
):
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
        return bool(self._obj.plain_is_zero())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    @override
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        other = cast("Self", other)

        if not self.sp.order_equal(other.sp):
            return False
        return self._obj.plain_is_equal(other._obj)

    def equals(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        other = cast("Self", other)

        aligned_lhs, aligned_rhs = _align_two(self, other)
        return aligned_lhs._obj.is_equal(aligned_rhs._obj)


@dataclass(frozen=True, eq=False)
class _NamedPwExpressionLike(_NamedExpressionLike[NamedExpressionLikeT_co]):
    ...


@dataclass(frozen=True, eq=False)
class Aff(_NamedExpressionLike[isl.Aff]):
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

        expr_space = zero.sp
        isl_zero = zero._obj
        for name, (dt, idx) in expr_space.name_to_dim.items():
            result[name] = Aff(
                isl_zero.set_coefficient_val(dt.as_isl(), idx, 1),
                expr_space)

        return result

    def set_coefficient(self, name: str, value: int) -> Aff:
        dt, idx = self.sp.name_to_dim[name]
        return Aff(self._obj.set_coefficient_val(dt.as_isl(), idx, value), self.sp)

    def as_pw_aff(self) -> PwAff:
        return PwAff(self._obj.to_pw_aff(), self.sp)


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
class PwAff(_NamedPwExpressionLike[isl.PwAff]):
    def gist(self, set: Set):
        self_a, set_a = _align_two(self, set)
        result = self_a._obj.gist(set_a._obj)
        return PwAff(result, self_a.sp)

    @staticmethod
    def from_space(space: Space) -> dict[str | Literal[0], PwAff]:
        """*space* is assumed to be a set-like space."""
        zero = Aff.zero_on_domain(space)
        result: dict[str | Literal[0], PwAff] = {0: zero.as_pw_aff()}

        expr_space = zero.sp
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
            rhs = PwAff(isl.PwAff.zero_on_domain(self._obj.space) + rhs, self.sp)
        elif isinstance(rhs, Aff):
            rhs = rhs.as_pw_aff()
        self_a, rhs_a = _align_two(self, rhs)
        from .set_like import Set
        return Set(func(self_a._obj, rhs_a._obj), self_a.sp.as_set_space())


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
class QPolynomial(_NamedExpressionLike[isl.QPolynomial]):
    """
    Name-aware wrapper around :class:`islpy.QPolynomial`.

    Construct instances with :func:`make_qpolynomial`.
    """


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
class PwQPolynomial(_NamedPwExpressionLike[isl.PwQPolynomial]):
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
class _NamedMultiExpressionLike(NamedIslObject[IslMultiExpressionLikeT_co]):
    pass


def _ordered_multi_dim_names(
    obj: isl.MultiAff | isl.PwMultiAff, dim_type: isl.dim_type
) -> tuple[str, ...]:
    space = obj.get_space()
    names: list[str] = []
    for dim in range(obj.dim(dim_type)):
        name = space.get_dim_name(dim_type, dim)
        if name is None:
            raise ValueError("duplicate or unnamed dimension found")
        names.append(name)
    return tuple(names)


def _make_multi_expression_parts(
    obj: isl.MultiAff | isl.PwMultiAff,
) -> tuple[Mapping[str, PwAff], NameToDim, DimTypeToNames]:
    output_names = _ordered_multi_dim_names(obj, isl.dim_type.out)

    parts: Mapping[str, PwAff] = constantdict({
        name: make_pw_aff(
            obj.get_at(dim).to_pw_aff()
            if isinstance(obj, isl.MultiAff)
            else obj.get_at(dim)
        )
        for dim, name in enumerate(output_names)
    })

    if parts:
        input_names = _ordered_part_dim_names(parts, isl.dim_type.in_)
        parameter_names = _ordered_part_dim_names(parts, isl.dim_type.param)
    else:
        input_names = _ordered_multi_dim_names(obj, isl.dim_type.in_)
        parameter_names = _ordered_multi_dim_names(obj, isl.dim_type.param)

    seen_names: set[str] = set()
    for name in (*output_names, *input_names, *parameter_names):
        if name in seen_names:
            raise ValueError(f"duplicate dimension name found: {name}")
        seen_names.add(name)

    all_names = [*output_names, *input_names, *parameter_names]
    name_to_dim: NameToDim = constantdict({
        name: dim for dim, name in enumerate(all_names)
    })

    dimtype_to_names: dict[isl.dim_type, frozenset[str]] = {}
    if input_names:
        dimtype_to_names[isl.dim_type.in_] = frozenset(input_names)
    if parameter_names:
        dimtype_to_names[isl.dim_type.param] = frozenset(parameter_names)

    return parts, name_to_dim, constantdict(dimtype_to_names)


def _ordered_part_dim_names(
    parts: Mapping[str, PwAff],
    dim_type: isl.dim_type,
) -> tuple[str, ...]:
    part_iter = iter(parts.items())
    _, first_part = next(part_iter)
    ordered_names = first_part.ordered_dim_names(dim_type)

    for output_name, part in part_iter:
        part_ordered_names = part.ordered_dim_names(dim_type)
        if part_ordered_names != ordered_names:
            raise ValueError(
                f"multi expression part '{output_name}' has inconsistent "
                f"{dim_type.name} dimension names"
            )

    return ordered_names


@dataclass(frozen=True, eq=False)
class PwMultiAff(_NamedMultiExpressionLike[isl.PwMultiAff]):

    def get_at(self, name: str) -> PwAff:
        """
        Return the output component named *name*.
        """
        if name not in self._names_for_dim_type(isl.dim_type.set):
            raise ValueError(f"unknown output name: {name}")
        return self._obj[name]

    @override
    def _reconstruct_isl_object(self) -> isl.PwMultiAff:
        space = self._multi_expression_space()
        if not self._obj:
            return isl.PwMultiAff.zero(space)

        pw_aff_list = isl.PwAffList.alloc(
            self._multi_expression_context(),
            len(self._obj),
        )
        for part in self._ordered_pw_aff_parts():
            pw_aff_list = pw_aff_list.add(part)

        return isl.PwMultiAff.from_multi_pw_aff(
            isl.MultiPwAff.from_pw_aff_list(space, pw_aff_list)
        )


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


@final
@dataclass(frozen=True, eq=False)
class MultiAff(_NamedMultiExpressionLike[isl.MultiAff]):
    def get_at(self, name: str) -> PwAff:
        """
        Return the output component named *name*.
        """
        if name not in self._names_for_dim_type(isl.dim_type.set):
            raise ValueError(f"unknown output name: {name}")
        return self._obj[name]

    @override
    def _reconstruct_isl_object(self) -> isl.MultiAff:
        space = self._multi_expression_space()
        if not self._obj:
            return isl.MultiAff.zero(space)

        aff_list = isl.AffList.alloc(
            self._multi_expression_context(),
            len(self._obj),
        )
        for part in self._ordered_pw_aff_parts():
            aff_list = aff_list.add(part.as_aff())

        return isl.MultiAff.from_aff_list(space, aff_list)


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

# }}}
