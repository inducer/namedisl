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
from typing import TYPE_CHECKING, Any, ClassVar, cast, overload

from constantdict import constantdict
from typing_extensions import Self, override

import islpy as isl

from .core import (
    DimType,
    IslMapLikeT,
    IslSetLikeT,
    IslSetOrMapLike,
    IslSetOrMapLikeT,
    NamedIslObject,
    NameToDim,
    Space,
    _align_and_apply_binary_op,
    _align_obj,
    _align_two,
    _dimtype_to_names,
    chunked_dims_by_type,
)
from .expression_like import PwAff, make_pw_aff, make_pw_multi_aff


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from namedisl import Aff, PwMultiAff


def _set_like_and(lhs: isl.Set, rhs: isl.Set) -> isl.Set:
    return cast("isl.Set", cast("Any", operator.and_)(lhs, rhs))


def _compare_set_like(
    lhs: _NamedIslSetOrMapLike[IslSetOrMapLike],
    rhs: _NamedIslSetOrMapLike[IslSetOrMapLike],
    op: Callable[[isl.Set, isl.Set], bool],
) -> bool:
    lhs_is_map = isinstance(lhs, _NamedIslMapLike)
    rhs_is_map = isinstance(rhs, _NamedIslMapLike)
    if lhs_is_map != rhs_is_map:
        return NotImplemented

    aligned_lhs, aligned_rhs = _align_two(lhs, rhs)

    assert isinstance(aligned_lhs._obj, isl.Set)
    assert isinstance(aligned_rhs._obj, isl.Set)
    return op(aligned_lhs._obj, aligned_rhs._obj)


@dataclass(frozen=True, eq=False)
class _NamedIslSetOrMapLike(NamedIslObject[IslSetOrMapLikeT], ABC):
    def eliminate(self: Self, names: Collection[str]) -> Self:
        obj = self._obj
        for dt, chunks in chunked_dims_by_type(
                names, self.sp.name_to_dim).items():
            for start, count in chunks:
                obj = cast("IslSetOrMapLikeT",
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
                obj = cast("IslSetOrMapLikeT",
                    obj.project_out(dt.as_isl(), start, count))

        return type(self)(obj, self.sp)

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

    def equate_dims(
        self,
        names: Sequence[tuple[str, str]]
    ) -> Self:
        obj = self._obj
        for lhs_name, rhs_name in names:
            if lhs_name != rhs_name:
                lhs_dim_id = self.sp.name_to_dim[lhs_name]
                rhs_dim_id = self.sp.name_to_dim[rhs_name]
                obj = cast("IslSetOrMapLikeT", obj.equate(
                    lhs_dim_id.dim_type.as_isl(),
                    lhs_dim_id.dim_index,
                    rhs_dim_id.dim_type.as_isl(),
                    rhs_dim_id.dim_index,
                ))

        return type(self)(obj, self.sp)

    def as_pw_multi_aff(self) -> PwMultiAff:
        """
        Reconstruct and convert this object to :class:`islpy.PwMultiAff`.
        """
        return make_pw_multi_aff(self.as_isl().as_pw_multi_aff())

    def gist(self, context: Self) -> Self:
        self_aligned, context_aligned = _align_two(self, context)
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
        if not isinstance(other, type(self)):
            return NotImplemented

        aligned_self, aligned_other = _align_two(self, other)
        return aligned_self._obj.plain_is_equal(aligned_other._obj)

    def __lt__(self, other: _NamedIslSetOrMapLike[IslSetOrMapLike]) -> bool:
        return _compare_set_like(self, other, isl.Set.is_strict_subset)

    def __le__(self, other: _NamedIslSetOrMapLike[IslSetOrMapLike]) -> bool:
        return _compare_set_like(self, other, isl.Set.is_subset)


@dataclass(frozen=True, eq=False)
class _NamedIslSetLike(_NamedIslSetOrMapLike[IslSetLikeT], ABC):
    active_dim_types: ClassVar[frozenset[DimType]] = frozenset({DimType.param, DimType.set})

    def dim_max(self, name: str) -> PwAff:
        dt, idx = self.sp.name_to_dim[name]
        if dt != DimType.set:
            raise ValueError("can only take max with respect to set dimensions")
        return make_pw_aff(self.as_isl().dim_max(idx))

    def dim_min(self, name: str) -> PwAff:
        dt, idx = self.sp.name_to_dim[name]
        if dt != DimType.set:
            raise ValueError("can only take min with respect to set dimensions")
        return make_pw_aff(self.as_isl().dim_min(idx))


@dataclass(frozen=True, eq=False)
class BasicSet(_NamedIslSetLike[isl.BasicSet]):
    def complement(self):
        return Set(self._obj.complement(), _dimtype_to_names=self._dimtype_to_names)

    def convex_hull(self):
        return self

    def add_equality_constraint(self, name: str, aff: Aff) -> BasicSet:
        pass


@overload
def make_basic_set(src: str, ctx: isl.Context | None = None) -> BasicSet: ...


@overload
def make_basic_set(src: isl.BasicSet) -> BasicSet: ...


def make_basic_set(src: str | isl.BasicSet, ctx: isl.Context | None = None) -> BasicSet:
    """
    Create a :class:`BasicSet` from isl syntax or an :class:`islpy.BasicSet`.
    """
    obj = isl.BasicSet(src, ctx) if isinstance(src, str) else src
    return BasicSet(
        obj,
        Space(_dimtype_to_names(obj, BasicSet.active_dim_types)))


@dataclass(frozen=True, eq=False)
class Set(_NamedIslSetLike[isl.Set]):
    """
    Name-aware wrapper around :class:`islpy.Set`.

    Construct instances with :func:`make_set`.
    """

    def complement(self):
        return Set(self._obj.complement(), self.sp)

    def convex_hull(self):
        return BasicSet(
            self._obj.convex_hull(), self.sp)

    def get_basic_sets(self) -> Sequence[BasicSet]:
        """
        Return the basic-set pieces of this set.
        """
        isl_obj = self.as_isl()

        bsets = isl_obj.get_basic_sets()
        return [make_basic_set(bset) for bset in bsets]


@overload
def make_set(src: str, ctx: isl.Context | None = None) -> Set: ...


@overload
def make_set(src: isl.Set) -> Set: ...


def make_set(src: isl.Set | str, ctx: isl.Context | None = None) -> Set:
    obj = isl.Set(src, ctx) if isinstance(src, str) else src
    return Set(obj, Space(_dimtype_to_names(obj, Set.active_dim_types)))


class _NamedIslMapLike(_NamedIslSetOrMapLike[IslMapLikeT]):
    active_dim_types: ClassVar[frozenset[DimType]] = frozenset(
        {DimType.param, DimType.in_, DimType.set})

    def _ordered_names(self, names: frozenset[str]) -> tuple[str, ...]:
        return tuple(sorted(names, key=self._name_to_dim.__getitem__))

    def _reject_surviving_name_collisions(
        self,
        collisions: frozenset[str],
    ) -> None:
        if collisions:
            raise ValueError(
                "composition would create duplicate surviving names: "
                + ", ".join(sorted(collisions))
            )

    def _reorder_interface(
        self, dim_type: isl.dim_type, ordered_names: tuple[str, ...]
    ) -> _NamedIslMapLike[PublicMapLikeT_co]:
        interface_names = (
            self.input_names if dim_type == isl.dim_type.in_ else self._output_names()
        )
        current_names = self._ordered_names(interface_names)
        if current_names == ordered_names:
            return self

        out_names = (
            ordered_names
            if dim_type == isl.dim_type.out
            else self._ordered_names(self._output_names())
        )
        in_names = (
            ordered_names
            if dim_type == isl.dim_type.in_
            else self._ordered_names(self.input_names)
        )
        param_names = self._ordered_names(self.parameter_names)

        ordering: NameToDim = constantdict({
            name: dim for dim, name in enumerate((*out_names, *in_names, *param_names))
        })

        return _align_obj(self, ordering, self._dimtype_to_names)

    def _validate_composable(
        self,
        lhs_dim_type: isl.dim_type,
        other: BasicMap | Map,
        rhs_dim_type: isl.dim_type,
    ) -> tuple[str, ...]:
        lhs_names = (
            self.input_names
            if lhs_dim_type == isl.dim_type.in_
            else self._output_names()
        )
        rhs_names = (
            other.input_names
            if rhs_dim_type == isl.dim_type.in_
            else other._output_names()
        )
        if lhs_names != rhs_names:
            raise ValueError("maps are not composable: interface names differ")
        return self._ordered_names(lhs_names)

    def intersect_domain(self, domain: BasicSet | Set) -> BasicMap | Map:
        """
        Return this map restricted to *domain*.
        """
        domain_obj = domain._reconstruct_isl_object()
        assert isinstance(domain_obj, isl.BasicSet | isl.Set)
        result = _apply_set_like_binary_op(
            self,
            self._map_with_universe(
                isl.dim_type.in_,
                domain_obj,
            ),
            _set_like_and,
        )
        assert isinstance(result, BasicMap | Map)
        return result

    def intersect_range(self, range_: BasicSet | Set) -> BasicMap | Map:
        """
        Return this map restricted to *range_*.
        """
        range_obj = range_._reconstruct_isl_object()
        assert isinstance(range_obj, isl.BasicSet | isl.Set)
        result = _apply_set_like_binary_op(
            self,
            self._map_with_universe(
                isl.dim_type.out,
                range_obj,
            ),
            _set_like_and,
        )
        assert isinstance(result, BasicMap | Map)
        return result

    def apply_range(self, other: BasicMap | Map) -> BasicMap | Map:
        """
        Compose this map with *other* on this map's range.

        The output names of this map must match the input names of *other*.
        """
        ordered_names = self._validate_composable(
            isl.dim_type.out, other, isl.dim_type.in_
        )
        reordered_other = other._reorder_interface(isl.dim_type.in_, ordered_names)
        assert isinstance(reordered_other, BasicMap | Map)
        self._reject_surviving_name_collisions(
            self.input_names & reordered_other._output_names()
        )
        result = self._map_obj().apply_range(reordered_other._map_obj())
        return self._wrap_map_result(result)

    def apply_domain(self, other: BasicMap | Map) -> BasicMap | Map:
        """
        Compose *other* with this map on this map's domain.

        The input names of this map must match the output names of *other*.
        """
        ordered_names = self._validate_composable(
            isl.dim_type.in_, other, isl.dim_type.out
        )
        reordered_other = other._reorder_interface(isl.dim_type.out, ordered_names)
        assert isinstance(reordered_other, BasicMap | Map)
        self._reject_surviving_name_collisions(
            reordered_other.input_names & self._output_names()
        )
        result = reordered_other._map_obj().apply_range(self._map_obj())
        return self._wrap_map_result(result)

    def reverse(self) -> BasicMap | Map:
        """
        Return the map with domain and range exchanged.
        """
        return self._wrap_map_result(self._map_obj().reverse())

    def domain(self) -> BasicSet | Set:
        """
        Return the domain as a named set.
        """
        domain = self._map_obj().domain()
        if isinstance(domain, isl.BasicSet):
            return make_basic_set(domain)
        return make_set(domain)

    def range(self) -> BasicSet | Set:
        """
        Return the range as a named set.
        """
        range_ = self._map_obj().range()
        if isinstance(range_, isl.BasicSet):
            return make_basic_set(range_)
        return make_set(range_)


@dataclass(frozen=True, eq=False)
class BasicMap(_NamedIslMapLike[isl.BasicMap]):
    """
    Name-aware wrapper around :class:`islpy.BasicMap`.

    Construct instances with :func:`make_basic_map`.
    """

    def complement(self):
        return Map(self._obj.complement(), _dimtype_to_names=self._dimtype_to_names)

    def convex_hull(self):
        return self

    @override
    def _map_obj(self) -> isl.BasicMap:
        obj = self._reconstruct_isl_object()
        assert isinstance(obj, isl.BasicMap)
        return obj

    @override
    def domain(self) -> BasicSet:
        return make_basic_set(self._map_obj().domain())

    @override
    def range(self) -> BasicSet:
        return make_basic_set(self._map_obj().range())

    @override
    def intersect_domain(self, domain: BasicSet | Set) -> BasicMap | Map:
        if isinstance(domain, BasicSet):
            range_space = self._map_obj().range().get_space()
            filter_map = make_basic_map(
                isl.BasicMap.from_domain_and_range(
                    domain._reconstruct_isl_object(), isl.BasicSet.universe(range_space)
                )
            )
            result = self & filter_map
            assert isinstance(result, BasicMap | Map)
            return result
        return super().intersect_domain(domain)

    @override
    def intersect_range(self, range_: BasicSet | Set) -> BasicMap | Map:
        if isinstance(range_, BasicSet):
            domain_space = self._map_obj().domain().get_space()
            filter_map = make_basic_map(
                isl.BasicMap.from_domain_and_range(
                    isl.BasicSet.universe(domain_space),
                    range_._reconstruct_isl_object(),
                )
            )
            result = self & filter_map
            assert isinstance(result, BasicMap | Map)
            return result
        return super().intersect_range(range_)


@overload
def make_basic_map(src: str, ctx: isl.Context | None = None) -> BasicMap: ...


@overload
def make_basic_map(src: isl.BasicMap) -> BasicMap: ...


def make_basic_map(src: str | isl.BasicMap, ctx: isl.Context | None = None) -> BasicMap:
    """
    Create a :class:`BasicMap` from isl syntax or an :class:`islpy.BasicMap`.
    """
    obj = isl.BasicMap(src, ctx) if isinstance(src, str) else src
    return BasicMap(
        obj,
        Space(_dimtype_to_names(obj, BasicSet.active_dim_types)))


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
class Map(_NamedIslMapLike[isl.Map]):
    """
    Name-aware wrapper around :class:`islpy.Map`.

    Construct instances with :func:`make_map`.
    """

    def complement(self):
        return Map(self._obj.complement(), self.sp)

    def convex_hull(self):
        return BasicMap(
            self._obj.convex_hull(), self.sp)


@overload
def make_map(src: str, ctx: isl.Context | None = None) -> Map: ...


@overload
def make_map(src: isl.Map) -> Map: ...


def make_map(src: str | isl.Map, ctx: isl.Context | None = None) -> Map:
    obj = isl.Map(src, ctx) if isinstance(src, str) else src
    return Map(obj, Space(_dimtype_to_names(obj, Map.active_dim_types)))
