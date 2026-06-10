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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, final, overload

from constantdict import constantdict
from typing_extensions import override

import islpy as isl

from .core import (
    DimTypeToNames,
    IslExpressionLikeT,
    IslScalarExpressionLike,
    NamedIslObject,
    NameToDim,
    _align_two,
    _make_named_object_pieces,
    _normalize_public_dim_type,
)


if TYPE_CHECKING:
    from collections.abc import Callable


PublicMultiExpressionLikeT = TypeVar(
    "PublicMultiExpressionLikeT",
    isl.MultiAff,
    isl.PwMultiAff,
)
NamedScalarExpressionLikeT_co = TypeVar(
    "NamedScalarExpressionLikeT_co",
    bound=IslScalarExpressionLike,
    covariant=True,
)


def _add_isl_expression(
    lhs: IslExpressionLikeT, rhs: IslExpressionLikeT | int
) -> IslExpressionLikeT:
    return cast("IslExpressionLikeT", cast("Any", operator.add)(lhs, rhs))


def _sub_isl_expression(
    lhs: IslExpressionLikeT, rhs: IslExpressionLikeT | int
) -> IslExpressionLikeT:
    return cast("IslExpressionLikeT", cast("Any", operator.sub)(lhs, rhs))


def _mul_isl_expression(
    lhs: IslExpressionLikeT, rhs: IslExpressionLikeT | int
) -> IslExpressionLikeT:
    return cast("IslExpressionLikeT", cast("Any", operator.mul)(lhs, rhs))


def _radd_isl_expression(lhs: int, rhs: IslExpressionLikeT) -> IslExpressionLikeT:
    return cast("IslExpressionLikeT", cast("Any", operator.add)(lhs, rhs))


def _rsub_isl_expression(lhs: int, rhs: IslExpressionLikeT) -> IslExpressionLikeT:
    return cast("IslExpressionLikeT", cast("Any", operator.sub)(lhs, rhs))


def _rmul_isl_expression(lhs: int, rhs: IslExpressionLikeT) -> IslExpressionLikeT:
    return cast("IslExpressionLikeT", cast("Any", operator.mul)(lhs, rhs))


def _explicitly_promote_isl_expressions(
    lhs: IslExpressionLikeT, rhs: IslExpressionLikeT
) -> tuple[IslExpressionLikeT, IslExpressionLikeT]:
    if isinstance(lhs, isl.Aff) and isinstance(rhs, isl.PwAff):
        return cast("IslExpressionLikeT", lhs.to_pw_aff()), rhs
    if isinstance(lhs, isl.PwAff) and isinstance(rhs, isl.Aff):
        return lhs, cast("IslExpressionLikeT", rhs.to_pw_aff())
    return lhs, rhs


# {{{ "base" named expression-likes (affs, pwaffs, qpolynomials, pwqpolynomials)

@dataclass(frozen=True, eq=False)
class _NamedExpressionLike(
    NamedIslObject[NamedScalarExpressionLikeT_co, NamedScalarExpressionLikeT_co]
):
    @overload
    def __add__(self: Aff, other: Aff | int) -> Aff: ...

    @overload
    def __add__(self: Aff, other: PwAff) -> PwAff: ...

    @overload
    def __add__(self: PwAff, other: Aff | PwAff | int) -> PwAff: ...

    @overload
    def __add__(self: QPolynomial, other: QPolynomial | int) -> QPolynomial: ...

    @overload
    def __add__(
        self: PwQPolynomial, other: PwQPolynomial | int
    ) -> PwQPolynomial: ...

    def __add__(
        self: _NamedExpressionLike[IslScalarExpressionLike],
        other: _NamedExpressionLike[IslScalarExpressionLike] | int,
    ) -> _NamedExpressionLike[IslScalarExpressionLike]:
        """
        Add another compatible named expression or an integer.
        """
        return _apply_expression_binary_op(self, other, _add_isl_expression)

    @overload
    def __radd__(self: Aff, other: int) -> Aff: ...

    @overload
    def __radd__(self: PwAff, other: int) -> PwAff: ...

    @overload
    def __radd__(self: QPolynomial, other: int) -> QPolynomial: ...

    @overload
    def __radd__(self: PwQPolynomial, other: int) -> PwQPolynomial: ...

    def __radd__(
        self: _NamedExpressionLike[IslScalarExpressionLike],
        other: int,
    ) -> _NamedExpressionLike[IslScalarExpressionLike]:
        """
        Add this expression to an integer.
        """
        return _apply_reflected_int_expression_op(self, other, _radd_isl_expression)

    @overload
    def __sub__(self: Aff, other: Aff | int) -> Aff: ...

    @overload
    def __sub__(self: Aff, other: PwAff) -> PwAff: ...

    @overload
    def __sub__(self: PwAff, other: Aff | PwAff | int) -> PwAff: ...

    @overload
    def __sub__(self: QPolynomial, other: QPolynomial | int) -> QPolynomial: ...

    @overload
    def __sub__(
        self: PwQPolynomial, other: PwQPolynomial | int
    ) -> PwQPolynomial: ...

    def __sub__(
        self: _NamedExpressionLike[IslScalarExpressionLike],
        other: _NamedExpressionLike[IslScalarExpressionLike] | int,
    ) -> _NamedExpressionLike[IslScalarExpressionLike]:
        """
        Subtract another compatible named expression or an integer.
        """
        return _apply_expression_binary_op(self, other, _sub_isl_expression)

    @overload
    def __rsub__(self: Aff, other: int) -> Aff: ...

    @overload
    def __rsub__(self: PwAff, other: int) -> PwAff: ...

    @overload
    def __rsub__(self: QPolynomial, other: int) -> QPolynomial: ...

    @overload
    def __rsub__(self: PwQPolynomial, other: int) -> PwQPolynomial: ...

    def __rsub__(
        self: _NamedExpressionLike[IslScalarExpressionLike],
        other: int,
    ) -> _NamedExpressionLike[IslScalarExpressionLike]:
        """
        Subtract this expression from an integer.
        """
        return _apply_reflected_int_expression_op(self, other, _rsub_isl_expression)

    @overload
    def __mul__(self: Aff, other: Aff | int) -> Aff: ...

    @overload
    def __mul__(self: Aff, other: PwAff) -> PwAff: ...

    @overload
    def __mul__(self: PwAff, other: Aff | PwAff | int) -> PwAff: ...

    @overload
    def __mul__(self: QPolynomial, other: QPolynomial | int) -> QPolynomial: ...

    @overload
    def __mul__(
        self: PwQPolynomial, other: PwQPolynomial | int
    ) -> PwQPolynomial: ...

    def __mul__(
        self: _NamedExpressionLike[IslScalarExpressionLike],
        other: _NamedExpressionLike[IslScalarExpressionLike] | int,
    ) -> _NamedExpressionLike[IslScalarExpressionLike]:
        """
        Multiply by another compatible named expression or an integer.
        """
        return _apply_expression_binary_op(self, other, _mul_isl_expression)

    @overload
    def __rmul__(self: Aff, other: int) -> Aff: ...

    @overload
    def __rmul__(self: PwAff, other: int) -> PwAff: ...

    @overload
    def __rmul__(self: QPolynomial, other: int) -> QPolynomial: ...

    @overload
    def __rmul__(self: PwQPolynomial, other: int) -> PwQPolynomial: ...

    def __rmul__(
        self: _NamedExpressionLike[IslScalarExpressionLike],
        other: int,
    ) -> _NamedExpressionLike[IslScalarExpressionLike]:
        """
        Multiply this expression by an integer.
        """
        return _apply_reflected_int_expression_op(self, other, _rmul_isl_expression)

    def is_zero(self) -> bool:
        """
        Return whether this expression is identically zero.
        """
        return bool(self._obj.is_zero())  # pyright: ignore[reportAttributeAccessIssue, reportUnknownArgumentType, reportUnknownMemberType]

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise NotImplementedError("Objects are not of the same type")
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class _NamedPwExpressionLike(_NamedExpressionLike[NamedScalarExpressionLikeT_co]):
    ...


@final
@dataclass(frozen=True, eq=False)
class Aff(_NamedExpressionLike[isl.Aff]):
    """
    Name-aware wrapper around :class:`islpy.Aff`.

    Construct instances with :func:`make_aff`.
    """

    _obj: isl.Aff

    @override
    def _reconstruct_isl_object(self) -> isl.Aff:
        obj = super()._reconstruct_isl_object()
        assert isinstance(obj, isl.Aff)
        return obj


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
    aff_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
    assert isinstance(aff_obj, isl.Aff)
    return Aff(aff_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class QPolynomial(_NamedExpressionLike[isl.QPolynomial]):
    """
    Name-aware wrapper around :class:`islpy.QPolynomial`.

    Construct instances with :func:`make_qpolynomial`.
    """

    _obj: isl.QPolynomial

    @override
    def _reconstruct_isl_object(self) -> isl.QPolynomial:
        obj = super()._reconstruct_isl_object()
        assert isinstance(obj, isl.QPolynomial)
        return obj


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
    obj = (
        isl.PwQPolynomial(src, ctx).get_pieces()[0][1] if isinstance(src, str)
        else src
    )

    qp_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
    assert isinstance(qp_obj, isl.QPolynomial)
    return QPolynomial(qp_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class PwAff(_NamedPwExpressionLike[isl.PwAff]):
    """
    Name-aware wrapper around :class:`islpy.PwAff`.

    Construct instances with :func:`make_pw_aff`.
    """

    _obj: isl.PwAff

    @override
    def _reconstruct_isl_object(self) -> isl.PwAff:
        obj = super()._reconstruct_isl_object()
        assert isinstance(obj, isl.PwAff)
        return obj


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
    pwaff_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
    assert isinstance(pwaff_obj, isl.PwAff)
    return PwAff(pwaff_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class PwQPolynomial(_NamedPwExpressionLike[isl.PwQPolynomial]):
    """
    Name-aware wrapper around :class:`islpy.PwQPolynomial`.

    Construct instances with :func:`make_pw_qpolynomial`.
    """

    _obj: isl.PwQPolynomial

    @override
    def _reconstruct_isl_object(self) -> isl.PwQPolynomial:
        obj = super()._reconstruct_isl_object()
        assert isinstance(obj, isl.PwQPolynomial)
        return obj


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
    pw_qp_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
    assert isinstance(pw_qp_obj, isl.PwQPolynomial)
    return PwQPolynomial(pw_qp_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


def _wrap_expression_result(
    result: IslExpressionLikeT,
    name_to_dim: NameToDim,
    dimtype_to_names: DimTypeToNames,
) -> _NamedExpressionLike[IslExpressionLikeT]:
    if isinstance(result, isl.Aff):
        return cast(
            "_NamedExpressionLike[IslExpressionLikeT]",
            Aff(result, name_to_dim, dimtype_to_names),  # pylint: disable=too-many-function-args
        )
    if isinstance(result, isl.PwAff):
        return cast(
            "_NamedExpressionLike[IslExpressionLikeT]",
            PwAff(result, name_to_dim, dimtype_to_names),  # pylint: disable=too-many-function-args
        )
    if isinstance(result, isl.QPolynomial):
        return cast(
            "_NamedExpressionLike[IslExpressionLikeT]",
            QPolynomial(result, name_to_dim, dimtype_to_names),  # pylint: disable=too-many-function-args
        )
    if isinstance(result, isl.PwQPolynomial):
        return cast(
            "_NamedExpressionLike[IslExpressionLikeT]",
            PwQPolynomial(result, name_to_dim, dimtype_to_names),  # pylint: disable=too-many-function-args
        )
    raise TypeError(f"unsupported expression result type: {type(result).__name__}")


@overload
def _apply_expression_binary_op(
    lhs: Aff,
    rhs: Aff | int,
    op: Callable[
        [IslScalarExpressionLike, IslScalarExpressionLike | int],
        IslScalarExpressionLike,
    ],
) -> Aff: ...


@overload
def _apply_expression_binary_op(
    lhs: Aff,
    rhs: PwAff,
    op: Callable[
        [IslScalarExpressionLike, IslScalarExpressionLike | int],
        IslScalarExpressionLike,
    ],
) -> PwAff: ...


@overload
def _apply_expression_binary_op(
    lhs: PwAff,
    rhs: Aff | PwAff | int,
    op: Callable[
        [IslScalarExpressionLike, IslScalarExpressionLike | int],
        IslScalarExpressionLike,
    ],
) -> PwAff: ...


@overload
def _apply_expression_binary_op(
    lhs: QPolynomial,
    rhs: QPolynomial | int,
    op: Callable[
        [IslScalarExpressionLike, IslScalarExpressionLike | int],
        IslScalarExpressionLike,
    ],
) -> QPolynomial: ...


@overload
def _apply_expression_binary_op(
    lhs: PwQPolynomial,
    rhs: PwQPolynomial | int,
    op: Callable[
        [IslScalarExpressionLike, IslScalarExpressionLike | int],
        IslScalarExpressionLike,
    ],
) -> PwQPolynomial: ...


@overload
def _apply_expression_binary_op(
    lhs: _NamedExpressionLike[IslScalarExpressionLike],
    rhs: _NamedExpressionLike[IslScalarExpressionLike] | int,
    op: Callable[
        [IslScalarExpressionLike, IslScalarExpressionLike | int],
        IslScalarExpressionLike,
    ],
) -> _NamedExpressionLike[IslScalarExpressionLike]: ...


def _apply_expression_binary_op(
    lhs: _NamedExpressionLike[IslScalarExpressionLike],
    rhs: _NamedExpressionLike[IslScalarExpressionLike] | int,
    op: Callable[
        [IslScalarExpressionLike, IslScalarExpressionLike | int],
        IslScalarExpressionLike,
    ],
) -> _NamedExpressionLike[IslScalarExpressionLike]:
    if isinstance(rhs, int):
        return _wrap_expression_result(
            op(lhs._obj, rhs),
            lhs._name_to_dim,
            lhs._dimtype_to_names,
        )

    lhs, rhs = _align_two(lhs, rhs)
    lhs_obj, rhs_obj = _explicitly_promote_isl_expressions(lhs._obj, rhs._obj)
    result = op(lhs_obj, rhs_obj)
    return _wrap_expression_result(
        result,
        lhs._name_to_dim,
        lhs._dimtype_to_names,
    )


@overload
def _apply_reflected_int_expression_op(
    expr: Aff,
    other: int,
    op: Callable[[int, IslScalarExpressionLike], IslScalarExpressionLike],
) -> Aff: ...


@overload
def _apply_reflected_int_expression_op(
    expr: PwAff,
    other: int,
    op: Callable[[int, IslScalarExpressionLike], IslScalarExpressionLike],
) -> PwAff: ...


@overload
def _apply_reflected_int_expression_op(
    expr: QPolynomial,
    other: int,
    op: Callable[[int, IslScalarExpressionLike], IslScalarExpressionLike],
) -> QPolynomial: ...


@overload
def _apply_reflected_int_expression_op(
    expr: PwQPolynomial,
    other: int,
    op: Callable[[int, IslScalarExpressionLike], IslScalarExpressionLike],
) -> PwQPolynomial: ...


@overload
def _apply_reflected_int_expression_op(
    expr: _NamedExpressionLike[IslScalarExpressionLike],
    other: int,
    op: Callable[[int, IslScalarExpressionLike], IslScalarExpressionLike],
) -> _NamedExpressionLike[IslScalarExpressionLike]: ...


def _apply_reflected_int_expression_op(
    expr: _NamedExpressionLike[IslScalarExpressionLike],
    other: int,
    op: Callable[[int, IslScalarExpressionLike], IslScalarExpressionLike],
) -> _NamedExpressionLike[IslScalarExpressionLike]:
    return _wrap_expression_result(
        op(other, expr._obj),
        expr._name_to_dim,
        expr._dimtype_to_names,
    )

# }}}


# {{{ multi expression-likes (multiaff, pwmultiaff)

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
class _NamedMultiExpressionLike(Generic[PublicMultiExpressionLikeT]):
    """
    Multi-expression components are stored directly as named :class:`PwAff`
    parts, keyed by output name.
    """

    _obj: Mapping[str, PwAff]
    _name_to_dim: NameToDim
    _dimtype_to_names: DimTypeToNames

    @property
    def _metadata_input_names(self) -> frozenset[str]:
        return self._dimtype_to_names.get(isl.dim_type.in_, frozenset())

    @property
    def _metadata_parameter_names(self) -> frozenset[str]:
        return self._dimtype_to_names.get(isl.dim_type.param, frozenset())

    def _names_for_dim_type(self, dim_type: isl.dim_type) -> frozenset[str]:
        dim_type = _normalize_public_dim_type(dim_type)
        if dim_type == isl.dim_type.param:
            return self.parameter_names
        if dim_type == isl.dim_type.in_:
            return self._metadata_input_names
        if dim_type == isl.dim_type.set:
            return frozenset(self._obj)
        raise ValueError(f"unsupported dim type: {dim_type}")

    def _ordered_names_for_dim_type(self, dim_type: isl.dim_type) -> tuple[str, ...]:
        names = self._names_for_dim_type(dim_type)
        return tuple(sorted(names, key=self._name_to_dim.__getitem__))

    @property
    def names(self) -> frozenset[str]:
        """
        All dimension names known to this object.
        """
        return self.output_names | self.input_names | self.parameter_names

    def dim_names(self, dim_type: isl.dim_type) -> frozenset[str]:
        """
        Return the names belonging to *dim_type*.
        """
        return self._names_for_dim_type(dim_type)

    def ordered_dim_names(self, dim_type: isl.dim_type) -> tuple[str, ...]:
        """
        Return names for *dim_type* in their current dimension order.
        """
        return self._ordered_names_for_dim_type(dim_type)

    @property
    def set_names(self) -> frozenset[str]:
        """
        Names of set dimensions.
        """
        return self._names_for_dim_type(isl.dim_type.set)

    @property
    def output_names(self) -> frozenset[str]:
        """
        Names of output dimensions.
        """
        return self._names_for_dim_type(isl.dim_type.out)

    @property
    def input_names(self) -> frozenset[str]:
        """
        Names of input dimensions.
        """
        return self._names_for_dim_type(isl.dim_type.in_)

    @property
    def parameter_names(self) -> frozenset[str]:
        """
        Names of parameter dimensions.
        """
        return self._metadata_parameter_names

    def dim(self, dim_type: isl.dim_type) -> int:
        """
        Return the number of dimensions of *dim_type*.
        """
        dim_type = _normalize_public_dim_type(dim_type)
        if dim_type in (isl.dim_type.set, isl.dim_type.in_, isl.dim_type.param):
            return len(self._names_for_dim_type(dim_type))
        return self._reconstruct_isl_object().dim(dim_type)

    def get_space(self) -> isl.Space:
        """
        Reconstruct and return the object's public isl space.
        """
        return self._reconstruct_isl_object().get_space()

    def get_isl_object(self) -> PublicMultiExpressionLikeT:
        """
        Reconstruct and return the wrapped public :mod:`islpy` object.
        """
        return self._reconstruct_isl_object()

    def _reconstruct_isl_object(self) -> PublicMultiExpressionLikeT:
        raise NotImplementedError

    def _multi_expression_context(self) -> isl.Context:
        if self._obj:
            return next(iter(self._obj.values()))._obj.get_ctx()
        return isl.DEFAULT_CONTEXT

    def _multi_expression_space(self) -> isl.Space:
        return isl.Space.create_from_names(
            self._multi_expression_context(),
            params=list(self.ordered_dim_names(isl.dim_type.param)),
            in_=list(self.ordered_dim_names(isl.dim_type.in_)),
            out=list(self.ordered_dim_names(isl.dim_type.out)),
        )

    def _ordered_pw_aff_parts(self) -> tuple[isl.PwAff, ...]:
        return tuple(
            self._obj[name]._reconstruct_isl_object()
            for name in self.ordered_dim_names(isl.dim_type.out)
        )


@final
@dataclass(frozen=True, eq=False)
class PwMultiAff(_NamedMultiExpressionLike[isl.PwMultiAff]):
    """
    Name-aware wrapper around :class:`islpy.PwMultiAff`.

    Construct instances with :func:`make_pw_multi_aff`.
    """

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
    """
    Create a :class:`PwMultiAff` from isl syntax or an :class:`islpy.PwMultiAff`.
    """

    obj = isl.PwMultiAff(src, ctx) if isinstance(src, str) else src
    pw_maff_obj, name_to_dim, dimtype_to_names = _make_multi_expression_parts(obj)
    return PwMultiAff(pw_maff_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class MultiAff(_NamedMultiExpressionLike[isl.MultiAff]):
    """
    Name-aware wrapper around :class:`islpy.MultiAff`.

    Construct instances with :func:`make_multi_aff`.
    """

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
    maff_obj, name_to_dim, dimtype_to_names = _make_multi_expression_parts(obj)
    return MultiAff(maff_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args

# }}}
