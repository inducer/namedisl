"""
.. currentmodule:: namedisl

Quasi-affine expression
-----------------------
.. autoclass:: Aff
.. autofunction:: make_aff
.. autofunction:: affs_from_domain_space

Piecewise quasi-affine expression
---------------------------------
.. autoclass:: PwAff
.. autofunction:: make_pw_aff
.. autofunction:: pw_affs_from_domain_space

Quasipolynomial
---------------
.. autoclass:: QPolynomial
.. autofunction:: make_qpolynomial

Piecewise quasipolynomial
-------------------------
.. autoclass:: PwQPolynomial
.. autofunction:: make_pw_qpolynomial

Vector-valued quasi-affine expression
-------------------------------------
.. autoclass:: MultiAff
.. autofunction:: make_multi_aff

Piecewise vector-valued quasi-affine expression
-----------------------------------------------
.. autoclass:: PwMultiAff
.. autofunction:: make_pw_multi_aff
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
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import Self, override

import islpy as isl

from .core import (
    DimType,
    IslAffLikeT_co,
    IslExpressionLikeT_co,
    IslObject,
    IslPolynomialLikeT_co,
    IslScalarExpressionLike,
    IslScalarExpressionLikeT,
    NamedIslObject,
    Space,
    align_expr_and_set,
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

class _NamedExpressionLike(NamedIslObject[IslExpressionLikeT_co]):
    __doc__ = """
    .. automethod:: __neg__
    .. automethod:: __add__
    .. automethod:: __radd__
    .. automethod:: __sub__
    .. automethod:: __rsub__
    .. automethod:: __mul__
    .. automethod:: __rmul__
    .. automethod:: __bool__
    .. automethod:: is_zero
    .. automethod:: plain_is_zero
    .. automethod:: equals
    """

    active_dim_types: ClassVar[frozenset[DimType]] = frozenset({
        DimType.param, DimType.in_})

    def __neg__(self):
        return type(self)(cast("IslExpressionLikeT_co", self._obj.neg()), self.space)

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

    def __bool__(self):
        raise RuntimeError("use plain_is_zero instead of truthiness")

    def plain_is_zero(self) -> bool:
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


class _NamedAffLike(_NamedExpressionLike[IslAffLikeT_co]):
    __doc__ = """
    .. automethod:: is_constant
    .. automethod:: gist
    .. automethod:: gist_params
    .. automethod:: __mod__
    """

    def is_constant(self) -> bool:
        return self._obj.is_cst()

    def gist(self, set: Set) -> Self:
        self_a, set_a = align_expr_and_set(self, set)
        if self_a._obj.get_domain_space().is_params():
            set_a_obj = set_a._obj.params()
        else:
            set_a_obj = set_a._obj

        return type(self)(
            cast("IslAffLikeT_co", self_a._obj.gist(set_a_obj)), self_a.space)

    def gist_params(self, set: Set) -> Self:
        self_a, set_a = align_expr_and_set(self, set)
        return type(self)(
            cast("IslAffLikeT_co", self_a._obj.gist_params(set_a._obj)), self_a.space)

    def __mod__(self, other: isl.Val | int) -> Self:
        return type(self)(cast("IslAffLikeT_co", self._obj.mod_val(other)), self.space)


class Aff(_NamedAffLike[isl.Aff]):
    __doc__ = f"""
    .. automethod:: zero_on_domain
    .. automethod:: set_coefficient
    .. automethod:: as_pw_aff

    .. autoattribute:: num_divs
    .. automethod:: get_div
    .. automethod:: get_div_coefficient

    .. automethod:: get_constant
    .. automethod:: get_denominator

    {_NamedAffLike.__doc__}
    {_NamedExpressionLike.__doc__}
    {NamedIslObject.__doc__}
    """

    _isl_type: ClassVar[type[IslObject]] = isl.Aff

    @staticmethod
    def zero_on_domain(space: Space) -> Aff:
        return Aff(
            isl.Aff.zero_on_domain(isl.LocalSpace.from_space(space.as_isl_set_space())),
            space.as_expr_space())

    def set_coefficient(self, name: str, value: int) -> Aff:
        dt, idx = self.space.name_to_dim[name]
        return Aff(self._obj.set_coefficient_val(dt.as_isl(), idx, value), self.space)

    def as_pw_aff(self) -> PwAff:
        return PwAff(self._obj.to_pw_aff(), self.space)

    @property
    def num_divs(self) -> int:
        return self._obj.dim(isl.dim_type.div)

    def get_div(self, index: int) -> Aff:
        return Aff(self._obj.get_div(index), self.space)

    def get_div_coefficient(self, index: int) -> isl.Val:
        return self._obj.get_coefficient_val(isl.dim_type.div, index)

    def get_coefficient(self, name: str) -> isl.Val:
        dt, idx = self.space.name_to_dim[name]
        return self._obj.get_coefficient_val(dt.as_isl(), idx)

    def get_constant(self) -> isl.Val:
        return self._obj.get_constant_val()

    def get_denominator(self) -> isl.Val:
        return self._obj.get_denominator_val()


@overload
def make_aff(src: str, ctx: isl.Context | None = None) -> Aff:
    ...


@overload
def make_aff(src: isl.Aff) -> Aff:
    ...


def make_aff(src: str | isl.Aff, ctx: isl.Context | None = None) -> Aff:
    obj = isl.Aff(src, ctx) if isinstance(src, str) else src
    return Aff(obj, Space.from_isl(obj, Aff.active_dim_types))


@dataclass(frozen=True)
class _AffMapping(Mapping[str | Literal[0], Aff]):
    expr_space: Space
    isl_zero: isl.Aff

    @override
    def __len__(self):
        return len(self.expr_space.name_to_dim) + 1

    @override
    def __iter__(self):
        yield 0
        yield from self.expr_space.name_to_dim.keys()

    @override
    def __getitem__(self, name: str | Literal[0]) -> Aff:
        if name == 0:
            return Aff(
            self.isl_zero,
            self.expr_space)

        dt, idx = self.expr_space.name_to_dim[name]
        return Aff(
            self.isl_zero.set_coefficient_val(dt.as_isl(), idx, 1),
            self.expr_space)


def affs_from_domain_space(space: Space) -> Mapping[str | Literal[0], Aff]:
    zero = Aff.zero_on_domain(space)
    return _AffMapping(zero.space, zero._obj)


class PwAff(_NamedAffLike[isl.PwAff]):
    __doc__ = f"""
    .. automethod:: from_piece_and_aff
    .. automethod:: zero_like_me
    .. autoattribute:: var_pw_affs
    .. automethod:: where
    .. automethod:: get_pieces
    .. automethod:: coalesce
    .. automethod:: eq_set
    .. automethod:: ne_set
    .. automethod:: ge_set
    .. automethod:: le_set
    .. automethod:: gt_set
    .. automethod:: lt_set
    .. automethod:: max
    .. automethod:: min
    .. automethod:: div
    .. automethod:: floor
    .. automethod:: get_aggregate_domain
    .. automethod:: union_max
    .. automethod:: union_min
    .. automethod:: union_add
    {_NamedAffLike.__doc__}
    {_NamedExpressionLike.__doc__}
    {NamedIslObject.__doc__}
    """

    _isl_type: ClassVar[type[IslObject]] = isl.PwAff

    @staticmethod
    def zero_on_domain(space: Space) -> PwAff:
        return PwAff(
            isl.PwAff.zero_on_domain(isl.LocalSpace.from_space(space.as_isl_set_space())),
            space.as_expr_space())

    @staticmethod
    def from_piece_and_aff(piece: Set, aff: Aff) -> PwAff:
        if not piece.space.order_equals(aff.space.as_set_space()):
            raise ValueError("spaces don't match")

        if aff.space.dim(DimType.in_):
            return PwAff(isl.PwAff.alloc(piece._obj, aff._obj), aff.space)
        else:
            return PwAff(isl.PwAff.alloc(piece._obj.params(), aff._obj), aff.space)

    def zero_like_me(self) -> PwAff:
        return PwAff(
            isl.PwAff.zero_on_domain(self._obj.get_domain_space()),
            self.space)

    @cached_property
    def var_pw_affs(self: PwAff) -> Mapping[str | Literal[0], PwAff]:
        r"""
        Returns a lazily-evaluated mapping from dimension names (or zero)
        to :class:`PwAff`\ s.

        .. note::

            Lazy evaluation means you do not pay for the creation of unused dimensions.
        """

        return _PwAffMapping(self.space, self._obj.get_domain_space())

    @override
    def plain_is_zero(self) -> bool:
        return self._obj.plain_is_equal(
            isl.PwAff.zero_on_domain(self._obj.get_domain_space()))

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
        rhs: int | PwAff
    ) -> Set:
        func = self._op_to_func[op]
        if isinstance(rhs, int):
            rhs = PwAff(
                isl.PwAff.zero_on_domain(self._obj.get_domain_space()) + rhs,
                self.space)
        self_a, rhs_a = align_two(self, rhs)
        from .set_like import Set
        res_set = func(self_a._obj, rhs_a._obj)
        return Set(
            res_set.from_params() if res_set.is_params() else res_set,
            self_a.space.as_set_space())

    def eq_set(self, rhs: int | PwAff) -> Set: return self.where("=", rhs)
    def ne_set(self, rhs: int | PwAff) -> Set: return self.where("!=", rhs)
    def ge_set(self, rhs: int | PwAff) -> Set: return self.where(">=", rhs)
    def le_set(self, rhs: int | PwAff) -> Set: return self.where("<=", rhs)
    def gt_set(self, rhs: int | PwAff) -> Set: return self.where(">", rhs)
    def lt_set(self, rhs: int | PwAff) -> Set: return self.where("<", rhs)

    def max(self, other: PwAff) -> PwAff:
        self_a, other_a = align_two(self, other)
        return PwAff(self_a._obj.max(other_a._obj), self_a.space)

    def min(self, other: PwAff) -> PwAff:
        self_a, other_a = align_two(self, other)
        return PwAff(self_a._obj.min(other_a._obj), self_a.space)

    def div(self, other: PwAff) -> PwAff:
        self_a, other_a = align_two(self, other)
        return PwAff(self_a._obj.div(other_a._obj), self_a.space)

    def floor(self) -> PwAff:
        return PwAff(self._obj.floor(), self.space)

    def get_pieces(self) -> list[tuple[Set, Aff]]:
        set_space = self.space.as_set_space()
        from .set_like import Set
        return [
            (Set(set.from_params() if set.is_params() else set,
                set_space),
                Aff(aff, self.space))
            for set, aff in self._obj.get_pieces()
        ]

    def coalesce(self) -> PwAff:
        return PwAff(self._obj.coalesce(), self.space)

    def get_aggregate_domain(self) -> Set:
        from .set_like import Set
        agg_domain = self._obj.get_aggregate_domain()
        return Set(
            agg_domain.from_params() if agg_domain.is_params() else agg_domain,
            self.space.as_set_space())

    def union_max(self, other: PwAff) -> PwAff:
        self_a, other_a = align_two(self, other)
        return PwAff(self_a._obj.union_max(other_a._obj), self_a.space)

    def union_min(self, other: PwAff) -> PwAff:
        self_a, other_a = align_two(self, other)
        return PwAff(self_a._obj.union_min(other_a._obj), self_a.space)

    def union_add(self, other: PwAff) -> PwAff:
        self_a, other_a = align_two(self, other)
        return PwAff(self_a._obj.union_add(other_a._obj), self_a.space)


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


@dataclass(frozen=True)
class _PwAffMapping(Mapping[str | Literal[0], PwAff]):
    expr_space: Space
    isl_domain_space: isl.Space

    @override
    def __len__(self):
        return len(self.expr_space.name_to_dim) + 1

    @override
    def __iter__(self):
        yield 0
        yield from self.expr_space.name_to_dim.keys()

    @override
    def __getitem__(self, name: str | Literal[0]) -> PwAff:
        if name == 0:
            return PwAff(
            isl.PwAff.zero_on_domain(self.isl_domain_space),
            self.expr_space)

        dt, idx = self.expr_space.name_to_dim[name]
        if dt == DimType.in_:
            dt = DimType.out
        return PwAff(
            isl.PwAff.var_on_domain(self.isl_domain_space, dt.as_isl(), idx),
            self.expr_space)


def pw_affs_from_domain_space(space: Space) -> Mapping[str | Literal[0], PwAff]:
    """This creates a lazily-evaluated mapping, i.e. you do not pay for the creation
    of unused dimensions.
    """
    return _PwAffMapping(space.as_expr_space(), space.as_isl_set_space())


class _NamedPolynomialLike(_NamedExpressionLike[IslPolynomialLikeT_co]):
    __doc__ = """
    .. automethod:: __pow__
    """

    def __pow__(self, other: int) -> Self:
        return type(self)(cast("IslPolynomialLikeT_co", self._obj ** other), self.space)


class QPolynomial(_NamedPolynomialLike[isl.QPolynomial]):
    __doc__ = f"""
    {_NamedPolynomialLike.__doc__}
    {_NamedExpressionLike.__doc__}
    {NamedIslObject.__doc__}
    """
    _isl_type: ClassVar[type[IslObject]] = isl.QPolynomial


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


class PwQPolynomial(_NamedPolynomialLike[isl.PwQPolynomial]):
    __doc__ = f"""
    .. automethod:: get_pieces
    {_NamedPolynomialLike.__doc__}
    {_NamedExpressionLike.__doc__}
    {NamedIslObject.__doc__}
    """

    _isl_type: ClassVar[type[IslObject]] = isl.PwQPolynomial

    def get_pieces(self) -> list[tuple[Set, QPolynomial]]:
        set_space = self.space.as_set_space()
        from .set_like import Set
        return [
            (Set(set, set_space), QPolynomial(qp, self.space))
            for set, qp in self._obj.get_pieces()
        ]


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

class MultiAff(_NamedExpressionLike[isl.MultiAff]):
    __doc__ = f"""
    .. automethod:: __getitem__
    {_NamedExpressionLike.__doc__}
    {NamedIslObject.__doc__}
    """
    _isl_type: ClassVar[type[IslObject]] = isl.MultiAff

    active_dim_types: ClassVar[frozenset[DimType]] = frozenset({
        DimType.param, DimType.in_, DimType.out})

    def __getitem__(self, name: str) -> Aff:
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


class PwMultiAff(_NamedExpressionLike[isl.PwMultiAff]):
    __doc__ = f"""
    .. automethod:: __getitem__
    .. automethod:: as_multi_aff
    {_NamedExpressionLike.__doc__}
    {NamedIslObject.__doc__}
    """

    _isl_type: ClassVar[type[IslObject]] = isl.PwMultiAff

    active_dim_types: ClassVar[frozenset[DimType]] = frozenset({
        DimType.param, DimType.in_, DimType.out})

    def __getitem__(self, name: str) -> PwAff:
        dt, idx = self.space.name_to_dim[name]
        if dt != DimType.out:
            raise ValueError(f"'{name}' does not name an output dimension")
        return PwAff(self._obj.get_at(idx), self.space.drop_dim_type(DimType.out))

    def as_multi_aff(self) -> MultiAff:
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
