"""
Name-aware set and map wrappers.

The classes in this module wrap :mod:`islpy` sets and maps while making
dimension names the primary way to address axes.  Internally, maps and sets are
stored as set-like isl objects with metadata that distinguishes output, input,
and parameter dimensions.
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
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, cast, overload

from constantdict import constantdict
from typing_extensions import Self, override

import islpy as isl

from .core import (
    DimType,
    IslMapLikeT,
    IslSetLikeT,
    IslSetOrMapLikeT,
    IslSetOrMapLikeT_co,
    IslUnbasicT_co,
    NamedIslObject,
    Space,
    _align_and_apply_binary_op,
    align_for_compostition,
    align_two,
    chunked_dims_by_type,
)
from .expression_like import make_pw_aff, make_pw_multi_aff


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from namedisl import Aff, PwMultiAff


def _compare_set_or_map_like(
    lhs: _NamedIslSetOrMapLike[IslSetOrMapLikeT],
    rhs: _NamedIslSetOrMapLike[IslSetOrMapLikeT],
    op: Callable[[IslSetOrMapLikeT, IslSetOrMapLikeT], bool],
) -> bool:
    if type(lhs) is not type(rhs):
        return NotImplemented

    aligned_lhs, aligned_rhs = align_two(lhs, rhs)

    return op(aligned_lhs._obj, aligned_rhs._obj)


@dataclass(frozen=True, eq=False)
class _NamedIslSetOrMapLike(NamedIslObject[IslSetOrMapLikeT_co], ABC):
    def eliminate(self: Self, names: Collection[str]) -> Self:
        obj = self._obj
        for dt, chunks in chunked_dims_by_type(
                names, self.sp.name_to_dim).items():
            for start, count in chunks:
                obj = cast("IslSetOrMapLikeT_co",
                    obj.eliminate(dt.as_isl(), start, count))

        return type(self)(obj, self.sp)

    def project_out(self: Self, names: str | Collection[str]) -> Self:
        new_dimtype_to_names = {
            dt: list(names) for dt, names in self.sp.dimtype_to_names.items()}
        obj = self._obj

        for dt, chunks in chunked_dims_by_type(
                names, self.sp.name_to_dim).items():
            for start, count in chunks[::-1]:
                del new_dimtype_to_names[dt][start:start+count]
                obj = cast("IslSetOrMapLikeT_co",
                    obj.project_out(dt.as_isl(), start, count))

        return type(self)(obj, Space(constantdict({
            dt: tuple(names)
            for dt, names in new_dimtype_to_names.items()
        })))

    def project_out_except(
        self: Self,
        names_to_keep: Collection[str],
    ) -> Self:
        if isinstance(names_to_keep, str):
            raise TypeError("names_to_keep must be a collection of str")

        names_to_project_out = [
            name for name in self.sp.name_to_dim if name not in names_to_keep
        ]

        return self.project_out(names_to_project_out)

    def gist(self, context: Self) -> Self:
        self_aligned, context_aligned = align_two(self, context)
        return type(self)(
            self_aligned._obj.gist(context_aligned._obj),
            self.sp,
        )

    def __and__(self, other: Self) -> Self:
        return cast("Self", _align_and_apply_binary_op(self, other, operator.and_))

    def __or__(self, other: Self) -> Self:
        return cast("Self", _align_and_apply_binary_op(self, other, operator.or_))

    def __sub__(self, other: Self) -> Self:
        return cast("Self", _align_and_apply_binary_op(self, other, operator.sub))

    @override
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        other = cast("Self", other)

        if not self.sp.order_equal(other.sp):
            return False
        if isinstance(self._obj, (isl.BasicSet, isl.BasicMap)):
            # these don't have plain_is_equal
            return self._obj.is_equal(other._obj)

        return self._obj.plain_is_equal(other._obj)

    def equals(self, other: Self) -> bool:
        return _compare_set_or_map_like(self, other, operator.eq)

    def __lt__(self, other: Self) -> bool:
        return _compare_set_or_map_like(self, other, operator.lt)

    def __le__(self, other: Self) -> bool:
        return _compare_set_or_map_like(self, other, operator.le)


@dataclass(frozen=True, eq=False)
class _NamedIslSetLike(_NamedIslSetOrMapLike[IslSetLikeT]):
    active_dim_types: ClassVar[frozenset[DimType]] = frozenset(
        {DimType.param, DimType.out})


@dataclass(frozen=True, eq=False)
class _NamedIslUnbasic(_NamedIslSetOrMapLike[IslUnbasicT_co]):
    def equate_dims(
        self,
        names: Sequence[tuple[str, str]]
    ) -> Self:
        obj = self._obj
        for lhs_name, rhs_name in names:
            if lhs_name != rhs_name:
                lhs_dim_id = self.sp.name_to_dim[lhs_name]
                rhs_dim_id = self.sp.name_to_dim[rhs_name]
                obj = cast("IslUnbasicT_co", obj.equate(
                    lhs_dim_id.dim_type.as_isl(),
                    lhs_dim_id.dim_index,
                    rhs_dim_id.dim_type.as_isl(),
                    rhs_dim_id.dim_index,
                ))

        return type(self)(obj, self.sp)

    def as_pw_multi_aff(self) -> PwMultiAff:
        return make_pw_multi_aff(self.as_isl().as_pw_multi_aff())


@dataclass(frozen=True, eq=False)
class BasicSet(_NamedIslSetLike[isl.BasicSet]):
    def add_eq_constraint(self, aff: Aff) -> BasicSet:
        if __debug__:  # noqa: SIM102
            if not self.sp.as_expr_space().order_equal(aff.sp):
                raise ValueError("spaces don't match")
        return BasicSet(
            self._obj.add_constraint(isl.Constraint.equality_from_aff(aff._obj)),
            self.sp)

    def add_ineq_constraint(self, aff: Aff) -> BasicSet:
        if __debug__:  # noqa: SIM102
            if not self.sp.as_expr_space().order_equal(aff.sp):
                raise ValueError("spaces don't match")
        return BasicSet(
            self._obj.add_constraint(isl.Constraint.inequality_from_aff(aff._obj)),
            self.sp)

    def affs(self) -> dict[str | Literal[0], Aff]:
        from .expression_like import Aff
        return Aff.from_space(self.sp)

    def as_set(self):
        return Set(self._obj.to_set(), self.sp)


@overload
def make_basic_set(src: str, ctx: isl.Context | None = None) -> BasicSet: ...


@overload
def make_basic_set(src: isl.BasicSet) -> BasicSet: ...


def make_basic_set(src: str | isl.BasicSet, ctx: isl.Context | None = None) -> BasicSet:
    obj = isl.BasicSet(src, ctx) if isinstance(src, str) else src
    return BasicSet(obj, Space.from_isl(obj, BasicSet.active_dim_types))


@dataclass(frozen=True, eq=False)
class Set(_NamedIslSetLike[isl.Set], _NamedIslUnbasic[isl.Set]):
    def complement(self):
        return Set(self._obj.complement(), self.sp)

    def convex_hull(self):
        return BasicSet(
            self._obj.convex_hull(), self.sp)

    def get_basic_sets(self):
        return [BasicSet(bs, self.sp) for bs in self._obj.get_basic_sets()]

    def dim_max(self, name: str):
        dt, idx = self.sp.name_to_dim[name]
        if dt != DimType.out:
            raise ValueError("can only take max with respect to set dimensions")
        return make_pw_aff(self.as_isl().dim_max(idx))

    def dim_min(self, name: str):
        dt, idx = self.sp.name_to_dim[name]
        if dt != DimType.out:
            raise ValueError("can only take min with respect to set dimensions")
        return make_pw_aff(self.as_isl().dim_min(idx))


@overload
def make_set(src: str, ctx: isl.Context | None = None) -> Set: ...


@overload
def make_set(src: isl.Set) -> Set: ...


def make_set(src: isl.Set | str, ctx: isl.Context | None = None) -> Set:
    obj = isl.Set(src, ctx) if isinstance(src, str) else src
    return Set(obj, Space.from_isl(obj, Set.active_dim_types))


class _NamedIslMapLike(_NamedIslSetOrMapLike[IslMapLikeT]):
    active_dim_types: ClassVar[frozenset[DimType]] = frozenset(
        {DimType.param, DimType.in_, DimType.out})

    def reverse(self) -> Self:
        return type(self)(
            cast("IslMapLikeT", self._obj.reverse()),
            self.sp.swap_dim_types(DimType.in_, DimType.out))


@dataclass(frozen=True, eq=False)
class BasicMap(_NamedIslMapLike[isl.BasicMap]):
    def domain(self):
        return BasicSet(self._obj.domain(), self.sp.drop_dim_type(DimType.out))

    def range(self):
        return BasicSet(self._obj.range(), self.sp.drop_dim_type(DimType.in_))

    def intersect_domain(self, domain: BasicSet) -> Self:
        self_a, domain_a = align_for_compostition(
            self, DimType.in_, domain, DimType.out)
        return type(self)(self_a._obj.intersect_domain(domain_a._obj), self_a.sp)

    def intersect_range(self, range: BasicSet) -> Self:
        self_a, range_a = align_for_compostition(
            self, DimType.out, range, DimType.out)
        return type(self)(self_a._obj.intersect_range(range_a._obj), self_a.sp)


@overload
def make_basic_map(src: str, ctx: isl.Context | None = None) -> BasicMap: ...


@overload
def make_basic_map(src: isl.BasicMap) -> BasicMap: ...


def make_basic_map(src: str | isl.BasicMap, ctx: isl.Context | None = None) -> BasicMap:
    obj = isl.BasicMap(src, ctx) if isinstance(src, str) else src
    return BasicMap(obj, Space.from_isl(obj, BasicMap.active_dim_types))


def make_map_from_domain_and_range(
    domain: BasicSet | Set, range_: BasicSet | Set
) -> BasicMap | Map:
    if isinstance(domain, BasicSet) and isinstance(range_, BasicSet):
        domain_obj = domain.as_isl()
        range_obj = range_.as_isl()
        assert isinstance(domain_obj, isl.BasicSet)
        assert isinstance(range_obj, isl.BasicSet)
        return make_basic_map(
            isl.BasicMap.from_domain_and_range(
                domain_obj,
                range_obj,
            )
        )

    domain_obj = domain.as_isl()
    range_obj = range_.as_isl()
    assert isinstance(domain_obj, isl.BasicSet | isl.Set)
    assert isinstance(range_obj, isl.BasicSet | isl.Set)
    return make_map(
        isl.Map.from_domain_and_range(
            domain_obj,
            range_obj,
        )
    )


@dataclass(frozen=True, eq=False)
class Map(_NamedIslMapLike[isl.Map], _NamedIslUnbasic[isl.Map]):
    def complement(self):
        return Map(self._obj.complement(), self.sp)

    def convex_hull(self):
        return BasicMap(
            self._obj.convex_hull(), self.sp)

    def get_basic_maps(self):
        return [BasicMap(bs, self.sp) for bs in self._obj.get_basic_maps()]

    def domain(self):
        return Set(self._obj.domain(), self.sp.drop_dim_type(DimType.out))

    def range(self):
        return Set(self._obj.range(), self.sp.drop_dim_type(DimType.in_))

    def intersect_domain(self, domain: Set) -> Self:
        self_a, domain_a = align_for_compostition(
            self, DimType.in_, domain, DimType.out)
        return type(self)(self_a._obj.intersect_domain(domain_a._obj), self_a.sp)

    def intersect_range(self, range: Set) -> Self:
        self_a, range_a = align_for_compostition(
            self, DimType.out, range, DimType.out)
        return type(self)(self_a._obj.intersect_range(range_a._obj), self_a.sp)

    def apply_range(self, other: Self) -> Self:
        self_a, other_a = align_for_compostition(self, DimType.out, other, DimType.in_)
        return type(self)(
            self._obj.apply_range(other_a._obj),
            Space(constantdict({
                DimType.param: self_a.sp.dimtype_to_names[DimType.param],
                DimType.in_: self_a.sp.dimtype_to_names[DimType.in_],
                DimType.out: other_a.sp.dimtype_to_names[DimType.out],
            })))

    def apply_domain(self, other: Self) -> Self:
        self_a, other_a = align_for_compostition(self, DimType.in_, other, DimType.out)
        return type(self)(
            self._obj.apply_domain(other_a._obj),
            Space(constantdict({
                DimType.param: self_a.sp.dimtype_to_names[DimType.param],
                DimType.in_: other_a.sp.dimtype_to_names[DimType.in_],
                DimType.out: self.sp.dimtype_to_names[DimType.out],
            })))


@overload
def make_map(src: str, ctx: isl.Context | None = None) -> Map: ...


@overload
def make_map(src: isl.Map) -> Map: ...


def make_map(src: str | isl.Map, ctx: isl.Context | None = None) -> Map:
    obj = isl.Map(src, ctx) if isinstance(src, str) else src
    return Map(obj, Space.from_isl(obj, Map.active_dim_types))
