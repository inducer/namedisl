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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeVar, cast, final, overload

from constantdict import constantdict
from typing_extensions import Self, override

import islpy as isl

from .core import (
    IslSetLike,
    NamedIslObject,
    NameToDim,
    _align_obj,
    _align_two,
    _find_contiguous_dim_chunks,
    _make_named_object_pieces,
)
from .expression_like import PwAff, make_pw_aff


PublicSetLikeT_co = TypeVar("PublicSetLikeT_co", bound=IslSetLike, covariant=True)
PublicMapLikeT_co = TypeVar(
    "PublicMapLikeT_co", isl.BasicMap, isl.Map, covariant=True
)


def _set_like_and(lhs: isl.Set, rhs: isl.Set) -> isl.Set:
    return cast("isl.Set", cast("Any", operator.and_)(lhs, rhs))


def _set_like_or(lhs: isl.Set, rhs: isl.Set) -> isl.Set:
    return cast("isl.Set", cast("Any", operator.or_)(lhs, rhs))


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence


@dataclass(frozen=True, eq=False)
class _NamedIslSetLike(NamedIslObject[isl.Set, PublicSetLikeT_co], ABC):
    """
    Represents set-like objects with parameter dimensions as a non-parameterized
    set. Names are organized as contiguous chunks of dimension types, i.e.
        [ (set names), (input names), (parameter names) ]
    """

    def complement(self: Self) -> Self:
        """
        Return the complement of this set-like object.
        """
        return replace(
            self,
            _obj=self._obj.complement(),
            _name_to_dim=self._name_to_dim,
            _dimtype_to_names=self._dimtype_to_names,
        )

    @overload
    def convex_hull(self: BasicMap | Map) -> BasicMap: ...

    @overload
    def convex_hull(self: BasicSet | Set) -> BasicSet: ...

    def convex_hull(self) -> BasicMap | BasicSet:
        """
        Return the convex hull as a basic set or basic map.
        """
        result = isl.Set.from_basic_set(self._obj.convex_hull())
        if isinstance(self, _NamedIslMapLike):
            return BasicMap(  # pylint: disable=too-many-function-args
                result,
                self._name_to_dim,
                self._dimtype_to_names,
            )

        return BasicSet(  # pylint: disable=too-many-function-args
            result,
            self._name_to_dim,
            self._dimtype_to_names,
        )

    def eliminate(self: Self, names_to_eliminate: str | Collection[str]) -> Self:
        """
        Eliminate constraints involving the named dimensions without removing them.
        """
        if isinstance(names_to_eliminate, str):
            names_to_eliminate = [names_to_eliminate]

        missing_names = [
            name for name in names_to_eliminate if name not in self.names
        ]
        if missing_names:
            raise ValueError(f"unknown names: {', '.join(missing_names)}")

        dims_to_eliminate = sorted(
            self._name_to_dim[name] for name in names_to_eliminate
        )

        contiguous_dim_chunks = _find_contiguous_dim_chunks(dims_to_eliminate)

        new_isl_obj = self._obj
        for start in sorted(contiguous_dim_chunks):
            new_isl_obj = new_isl_obj.eliminate(
                isl.dim_type.set, start, contiguous_dim_chunks[start]
            )

        return replace(
            self,
            _obj=new_isl_obj,
            _name_to_dim=self._name_to_dim,  # NOTE: no dims removed by eliminate
            _dimtype_to_names=self._dimtype_to_names,
        )

    def equate_dims(
        self,
        names: Sequence[tuple[str, str]]
    ) -> Self:
        obj = self._obj
        for lhs_name, rhs_name in names:
            if lhs_name != rhs_name:
                lhs_dim_id = self._name_to_dim[lhs_name]
                rhs_dim_id = self._name_to_dim[rhs_name]
                obj = obj.equate(
                    lhs_dim_id.dim_type.as_isl(),
                    lhs_dim_id.dim_index,
                    rhs_dim_id.dim_type.as_isl(),
                    rhs_dim_id.dim_index,
                )

        return type(self)(
            cast("InternalIslObjectT_co", obj),
            _dimtype_to_names=self._dimtype_to_names,
        )

    @overload
    def gist(
        self: BasicMap, context: _NamedIslSetLike[IslSetLike]
    ) -> BasicMap | Map: ...

    @overload
    def gist(self: Map, context: _NamedIslSetLike[IslSetLike]) -> Map: ...

    @overload
    def gist(
        self: BasicSet, context: _NamedIslSetLike[IslSetLike]
    ) -> BasicSet | Set: ...

    @overload
    def gist(self: Set, context: _NamedIslSetLike[IslSetLike]) -> Set: ...

    def gist(
        self, context: _NamedIslSetLike[IslSetLike]
    ) -> _NamedIslSetLike[IslSetLike]:
        """
        Simplify this object under the assumptions described by *context*.
        """
        self_aligned, context_aligned = _align_two(self, context)
        result = self_aligned._obj.gist(context_aligned._obj)

        if isinstance(self, BasicMap):
            result_type = BasicMap if result.n_basic_set() == 1 else Map
        elif isinstance(self, Map):
            result_type = Map
        elif isinstance(self, BasicSet):
            result_type = BasicSet if result.n_basic_set() == 1 else Set
        else:
            result_type = Set

        return result_type(
            result,
            _dimtype_to_names=self_aligned._dimtype_to_names,
        )

    def project_out(self: Self, names_to_project_out: str | Collection[str]) -> Self:
        """
        Return a copy with the named dimensions projected out.
        """

        if isinstance(names_to_project_out, str):
            names_to_project_out = [names_to_project_out]

        missing_names = [
            name for name in names_to_project_out if name not in self.names
        ]
        if missing_names:
            raise ValueError(f"unknown names: {', '.join(missing_names)}")

        names_to_remove = set(names_to_project_out)

        dims_to_remove = sorted(self._name_to_dim[name] for name in names_to_remove)

        new_isl_obj = self._obj
        contiguous_dim_chunks = _find_contiguous_dim_chunks(dims_to_remove)
        for start in sorted(contiguous_dim_chunks, reverse=True):
            new_isl_obj = new_isl_obj.project_out(
                isl.dim_type.set, start, contiguous_dim_chunks[start]
            )

        new_name_to_dim: NameToDim = {}
        for name, dim in self._name_to_dim.items():
            if name in names_to_remove:
                continue

            shift = 0
            for removed_dim in dims_to_remove:
                if removed_dim < dim:
                    shift += 1
                else:
                    break

            new_name_to_dim[name] = dim - shift

        new_type_to_names = constantdict({
            dt: self._dimtype_to_names[dt] - frozenset(names_to_remove)
            for dt in self._dimtype_to_names
        })

        return replace(
            self,
            _obj=new_isl_obj,
            _name_to_dim=constantdict(new_name_to_dim),
            _dimtype_to_names=new_type_to_names,
        )

    def project_out_except(
        self: Self,
        names_to_keep: str | Collection[str],
    ) -> Self:
        """
        Project out every dimension except those named in *names_to_keep*.
        """

        if isinstance(names_to_keep, str):
            names_to_keep = [names_to_keep] if names_to_keep else []

        names_to_project_out = [
            name for name in self._name_to_dim if name not in names_to_keep
        ]

        return self.project_out(names_to_project_out)

    def dim_max(self, name: str) -> PwAff:
        """
        Return the parametric maximum of the named dimension.
        """
        obj, dim = self._dim_bound_object_and_dim(name)
        return make_pw_aff(obj.dim_max(dim))

    def dim_min(self, name: str) -> PwAff:
        """
        Return the parametric minimum of the named dimension.
        """
        obj, dim = self._dim_bound_object_and_dim(name)
        return make_pw_aff(obj.dim_min(dim))

    def _dim_bound_object_and_dim(
        self, name: str
    ) -> tuple[isl.BasicSet | isl.Set, int]:
        if name not in self.names:
            raise ValueError(f"unknown name: {name}")
        if name in self.parameter_names:
            raise ValueError(f"cannot compute a bound for parameter: {name}")

        if isinstance(self, _NamedIslMapLike):
            bound_set = self.domain() if name in self.input_names else self.range()
            obj = bound_set._reconstruct_isl_object()
            assert isinstance(obj, isl.BasicSet | isl.Set)
            return obj, bound_set._name_to_dim[name]

        obj = self._reconstruct_isl_object()
        assert isinstance(obj, isl.BasicSet | isl.Set)
        return obj, self._name_to_dim[name]

    def is_empty(self) -> bool:
        """
        Return whether this object contains no integer points.
        """
        return bool(self._obj.is_empty())

    def as_pw_multi_aff(self) -> isl.PwMultiAff:
        """
        Reconstruct and convert this object to :class:`islpy.PwMultiAff`.
        """
        obj = self._reconstruct_isl_object()
        assert isinstance(obj, isl.Set | isl.Map)
        return obj.as_pw_multi_aff()

    @override
    def dim(self, dim_type: isl.dim_type) -> int:
        if dim_type == isl.dim_type.out:
            dim_type = isl.dim_type.set
        return super().dim(dim_type)

    @overload
    def __and__(self: BasicMap, other: BasicMap | Map) -> BasicMap | Map: ...

    @overload
    def __and__(self: Map, other: BasicMap | Map) -> Map: ...

    @overload
    def __and__(self: BasicSet, other: BasicSet | Set) -> BasicSet | Set: ...

    @overload
    def __and__(self: Set, other: BasicSet | Set) -> Set: ...

    def __and__(
        self, other: _NamedIslSetLike[IslSetLike]
    ) -> _NamedIslSetLike[IslSetLike]:
        """
        Return the intersection of two compatible named set-like objects.
        """
        return _apply_set_like_binary_op(self, other, _set_like_and)

    @overload
    def __or__(self: BasicMap, other: BasicMap | Map) -> BasicMap | Map: ...

    @overload
    def __or__(self: Map, other: BasicMap | Map) -> Map: ...

    @overload
    def __or__(self: BasicSet, other: BasicSet | Set) -> BasicSet | Set: ...

    @overload
    def __or__(self: Set, other: BasicSet | Set) -> Set: ...

    def __or__(
        self, other: _NamedIslSetLike[IslSetLike]
    ) -> _NamedIslSetLike[IslSetLike]:
        """
        Return the union of two compatible named set-like objects.
        """
        return _apply_set_like_binary_op(self, other, _set_like_or)

    @overload
    def __sub__(self: BasicMap, other: BasicMap | Map) -> BasicMap | Map: ...

    @overload
    def __sub__(self: Map, other: BasicMap | Map) -> Map: ...

    @overload
    def __sub__(self: BasicSet, other: BasicSet | Set) -> BasicSet | Set: ...

    @overload
    def __sub__(self: Set, other: BasicSet | Set) -> Set: ...

    def __sub__(
        self, other: _NamedIslSetLike[IslSetLike]
    ) -> _NamedIslSetLike[IslSetLike]:
        """
        Return the set difference with *other* removed.
        """
        return _align_and_apply_binary_op(self, other, operator.sub)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise NotImplementedError("Objects are not of the same type")

        aligned_self, aligned_other = _align_two(self, other)

        # FIXME: type checker complains because it's not clear whether the
        # underlying object after alignment is an isl.Set
        assert isinstance(aligned_self._obj, isl.Set)
        assert isinstance(aligned_other._obj, isl.Set)
        return aligned_self._obj.plain_is_equal(aligned_other._obj)

    def __lt__(self, other: _NamedIslSetLike[IslSetLike]) -> bool:
        """
        Return whether this object is a strict subset of *other*.
        """
        return _compare_set_like(self, other, isl.Set.is_strict_subset)

    def __le__(self, other: _NamedIslSetLike[IslSetLike]) -> bool:
        """
        Return whether this object is a subset of *other*.
        """
        return _compare_set_like(self, other, isl.Set.is_subset)

    def __gt__(self, other: _NamedIslSetLike[IslSetLike]) -> bool:
        """
        Return whether this object is a strict superset of *other*.
        """
        return _compare_set_like(other, self, isl.Set.is_strict_subset)

    def __ge__(self, other: _NamedIslSetLike[IslSetLike]) -> bool:
        """
        Return whether this object is a superset of *other*.
        """
        return _compare_set_like(other, self, isl.Set.is_subset)


@final
@dataclass(frozen=True, eq=False)
class BasicSet(_NamedIslSetLike[isl.BasicSet]):
    pass
    # TODO: add_constraint?


@overload
def make_basic_set(src: str, ctx: isl.Context | None = None) -> BasicSet: ...


@overload
def make_basic_set(src: isl.BasicSet) -> BasicSet: ...


def make_basic_set(src: str | isl.BasicSet, ctx: isl.Context | None = None) -> BasicSet:
    """
    Create a :class:`BasicSet` from isl syntax or an :class:`islpy.BasicSet`.
    """
    obj = isl.BasicSet(src, ctx) if isinstance(src, str) else src
    set_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
    assert isinstance(set_obj, isl.Set)
    return BasicSet(set_obj, name_to_dim, dimtype_to_names)


@final
@dataclass(frozen=True, eq=False)
class Set(_NamedIslSetLike[isl.Set]):
    """
    Name-aware wrapper around :class:`islpy.Set`.

    Construct instances with :func:`make_set`.
    """

    @override
    def add_input_names(self, names_to_add: Collection[str]) -> Set:
        raise NotImplementedError

    @override
    def _reconstruct_isl_object(self) -> isl.Set:
        obj = super()._reconstruct_isl_object()
        assert isinstance(obj, isl.Set)
        return obj

    def get_basic_sets(self) -> Sequence[BasicSet]:
        """
        Return the basic-set pieces of this set.
        """
        isl_obj = self._reconstruct_isl_object()

        bsets = isl_obj.get_basic_sets()
        return [make_basic_set(bset) for bset in bsets]


def _compare_set_like(
    lhs: _NamedIslSetLike[IslSetLike],
    rhs: _NamedIslSetLike[IslSetLike],
    op: Callable[[isl.Set, isl.Set], bool],
) -> bool:
    lhs_is_map = isinstance(lhs, _NamedIslMapLike)
    rhs_is_map = isinstance(rhs, _NamedIslMapLike)
    if lhs_is_map != rhs_is_map:
        raise TypeError("Cannot compare set-like and map-like objects")

    aligned_lhs, aligned_rhs = _align_two(lhs, rhs)

    assert isinstance(aligned_lhs._obj, isl.Set)
    assert isinstance(aligned_rhs._obj, isl.Set)
    return op(aligned_lhs._obj, aligned_rhs._obj)


class _NamedIslMapLike(_NamedIslSetLike[PublicMapLikeT_co]):
    @override
    def _reconstruct_isl_object(self) -> PublicMapLikeT_co:
        obj = super()._reconstruct_isl_object()
        if isinstance(obj, isl.Set):
            return cast(
                "PublicMapLikeT_co",
                isl.Map.from_domain_and_range(isl.Set("{ [] }"), obj),
            )
        assert isinstance(obj, isl.BasicMap | isl.Map)
        return cast("PublicMapLikeT_co", obj)

    def _output_names(self) -> frozenset[str]:
        return frozenset(self._name_to_dim) - self.input_names - self.parameter_names

    def _map_obj(self) -> isl.BasicMap | isl.Map:
        return self._reconstruct_isl_object()

    @staticmethod
    def _wrap_map_result(result: isl.BasicMap | isl.Map) -> BasicMap | Map:
        if isinstance(result, isl.BasicMap):
            return make_basic_map(result)
        return make_map(result)

    def _map_with_universe(
        self, dim_type: isl.dim_type, set_obj: isl.BasicSet | isl.Set
    ) -> BasicMap | Map:
        map_obj = self._map_obj()
        if dim_type == isl.dim_type.in_:
            universe = isl.Set.universe(map_obj.range().get_space())
            return make_map(isl.Map.from_domain_and_range(set_obj, universe))
        if dim_type == isl.dim_type.out:
            universe = isl.Set.universe(map_obj.domain().get_space())
            return make_map(isl.Map.from_domain_and_range(universe, set_obj))
        raise ValueError(f"unsupported dim type: {dim_type}")

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


@overload
def make_set(src: str, ctx: isl.Context | None = None) -> Set: ...


@overload
def make_set(src: isl.Set) -> Set: ...


def make_set(src: isl.Set | str, ctx: isl.Context | None = None) -> Set:
    """
    Create a :class:`Set` from isl syntax or an :class:`islpy.Set`.
    """
    obj = isl.Set(src, ctx) if isinstance(src, str) else src
    set_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
    assert isinstance(set_obj, isl.Set)
    return Set(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class BasicMap(_NamedIslMapLike[isl.BasicMap]):
    """
    Name-aware wrapper around :class:`islpy.BasicMap`.

    Construct instances with :func:`make_basic_map`.
    """

    @classmethod
    def empty(cls, space: isl.Space) -> BasicMap:
        """
        Return an empty :class:`BasicMap` in *space*.
        """
        obj = isl.BasicMap.empty(space)
        set_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
        assert isinstance(set_obj, isl.Set)
        return cls(  # pylint: disable=too-many-function-args
            set_obj,
            name_to_dim,
            dimtype_to_names,
        )

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

    @override
    def _reconstruct_isl_object(self) -> isl.BasicMap:
        obj = super()._reconstruct_isl_object()

        if isinstance(obj, isl.Map) and obj.is_empty():
            return isl.BasicMap.empty(obj.get_space())

        if not isinstance(obj, isl.Map) or obj.n_basic_map() != 1:
            raise ValueError(
                "Cannot reconstruct an isl.BasicMap from anything other than "
                "an isl.Map containing only a single isl.BasicMap."
            )

        return obj.get_basic_maps()[0]


@overload
def make_basic_map(src: str, ctx: isl.Context | None = None) -> BasicMap: ...


@overload
def make_basic_map(src: isl.BasicMap) -> BasicMap: ...


def make_basic_map(src: str | isl.BasicMap, ctx: isl.Context | None = None) -> BasicMap:
    """
    Create a :class:`BasicMap` from isl syntax or an :class:`islpy.BasicMap`.
    """
    obj = isl.BasicMap(src, ctx) if isinstance(src, str) else src
    set_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
    assert isinstance(set_obj, isl.Set)
    return BasicMap(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


def make_map_from_domain_and_range(
    domain: BasicSet | Set, range_: BasicSet | Set
) -> BasicMap | Map:
    """
    Create a named map from a named *domain* and named *range_*.

    A :class:`BasicMap` is returned when both inputs are basic sets; otherwise a
    :class:`Map` is returned.
    """
    if isinstance(domain, BasicSet) and isinstance(range_, BasicSet):
        domain_obj = domain._reconstruct_isl_object()
        range_obj = range_._reconstruct_isl_object()
        assert isinstance(domain_obj, isl.BasicSet)
        assert isinstance(range_obj, isl.BasicSet)
        return make_basic_map(
            isl.BasicMap.from_domain_and_range(
                domain_obj,
                range_obj,
            )
        )

    domain_obj = domain._reconstruct_isl_object()
    range_obj = range_._reconstruct_isl_object()
    assert isinstance(domain_obj, isl.BasicSet | isl.Set)
    assert isinstance(range_obj, isl.BasicSet | isl.Set)
    return make_map(
        isl.Map.from_domain_and_range(
            domain_obj,
            range_obj,
        )
    )


@final
@dataclass(frozen=True, eq=False)
class Map(_NamedIslMapLike[isl.Map]):
    """
    Name-aware wrapper around :class:`islpy.Map`.

    Construct instances with :func:`make_map`.
    """

    @classmethod
    def empty(cls, space: isl.Space) -> Map:
        """
        Return an empty :class:`Map` in *space*.
        """
        return make_map(isl.Map.empty(space))

    @override
    def _reconstruct_isl_object(self) -> isl.Map:
        obj = super()._reconstruct_isl_object()
        assert isinstance(obj, isl.Map)
        return obj


@overload
def make_map(src: str, ctx: isl.Context | None = None) -> Map: ...


@overload
def make_map(src: isl.Map) -> Map: ...


def make_map(src: str | isl.Map, ctx: isl.Context | None = None) -> Map:
    """
    Create a :class:`Map` from isl syntax or an :class:`islpy.Map`.
    """
    obj = isl.Map(src, ctx) if isinstance(src, str) else src
    set_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
    assert isinstance(set_obj, isl.Set)
    return Map(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args
