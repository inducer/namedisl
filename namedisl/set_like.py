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
from typing import TYPE_CHECKING, final, overload

from constantdict import constantdict
from typing_extensions import Self, override

import islpy as isl

from .core import (
    NamedIslObject,
    NameToDim,
    _align_obj,
    _align_two,
    _find_contiguous_dim_chunks,
    _make_named_object_pieces,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence


@dataclass(frozen=True, eq=False)
class _NamedIslSetLike(NamedIslObject[isl.Set], ABC):
    """
    Represents set-like objects with parameter dimensions as a non-parameterized
    set. Names are organized as contiguous chunks of dimension types, i.e.
        [ (set names), (input names), (parameter names) ]
    """

    def complement(self: Self) -> Self:
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
        obj = self._reconstruct_isl_object()
        assert isinstance(obj, isl.BasicMap | isl.Map | isl.BasicSet | isl.Set)

        if isinstance(obj, isl.BasicMap):
            return make_basic_map(obj.to_map().convex_hull())

        if isinstance(obj, isl.Map):
            return make_basic_map(obj.convex_hull())

        if isinstance(obj, isl.BasicSet):
            return make_basic_set(obj.to_set().convex_hull())

        return make_basic_set(obj.convex_hull())

    def eliminate(self: Self, names_to_eliminate: str | Collection[str]) -> Self:
        if isinstance(names_to_eliminate, str):
            names_to_eliminate = [names_to_eliminate]

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

    def add_constraint(
        self: Self,
        constraints: str | Collection[str],
    ) -> Self:
        if isinstance(constraints, str):
            constraints = [constraints]
        else:
            constraints = list(constraints)

        if not constraints:
            return self

        ordered_names = tuple(
            sorted(self._name_to_dim, key=self._name_to_dim.__getitem__)
        )
        constraint_text = " and ".join(f"({constraint})" for constraint in constraints)
        constraint_src = f"{{ [{', '.join(ordered_names)}] : {constraint_text} }}"

        try:
            constraint_obj = isl.Set(constraint_src)
        except isl.Error as exc:
            raise ValueError(
                f"invalid constraint for names {ordered_names}: {constraint_text}"
            ) from exc

        constraint_obj = constraint_obj.remove_redundancies()
        constraint_set, constraint_name_to_dim, _ = _make_named_object_pieces(
            constraint_obj
        )
        assert isinstance(constraint_set, isl.Set)

        if constraint_name_to_dim != self._name_to_dim:
            constraint_set = _align_obj(
                Set(constraint_set, constraint_name_to_dim, self._dimtype_to_names),
                self._name_to_dim,
                self._dimtype_to_names,
            )._obj
            assert isinstance(constraint_set, isl.Set)

        return replace(
            self,
            _obj=self._obj.intersect(constraint_set),
            _name_to_dim=self._name_to_dim,
            _dimtype_to_names=self._dimtype_to_names,
        )

    @overload
    def gist(self: BasicMap, context: _NamedIslSetLike) -> BasicMap | Map: ...

    @overload
    def gist(self: Map, context: _NamedIslSetLike) -> Map: ...

    @overload
    def gist(self: BasicSet, context: _NamedIslSetLike) -> BasicSet | Set: ...

    @overload
    def gist(self: Set, context: _NamedIslSetLike) -> Set: ...

    def gist(self, context: _NamedIslSetLike) -> _NamedIslSetLike:
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
            result, self_aligned._name_to_dim, self_aligned._dimtype_to_names
        )

    def project_out(self: Self, names_to_project_out: str | Collection[str]) -> Self:

        if isinstance(names_to_project_out, str):
            names_to_project_out = [names_to_project_out]

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

        if isinstance(names_to_keep, str):
            names_to_keep = [names_to_keep] if names_to_keep else []

        names_to_project_out = [
            name for name in self._name_to_dim if name not in names_to_keep
        ]

        return self.project_out(names_to_project_out)

    def dim_max(self, name: str) -> isl.PwAff:
        return self._obj.dim_max(self._name_to_dim[name])

    def dim_min(self, name: str) -> isl.PwAff:
        return self._obj.dim_min(self._name_to_dim[name])

    def is_empty(self) -> bool:
        obj = self._reconstruct_isl_object()
        assert isinstance(obj, isl.Set | isl.Map)
        return bool(obj.is_empty())

    def as_pw_multi_aff(self) -> isl.PwMultiAff:
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

    def __and__(self, other: _NamedIslSetLike) -> _NamedIslSetLike:
        return _apply_set_like_binary_op(self, other, operator.and_)

    @overload
    def __or__(self: BasicMap, other: BasicMap | Map) -> BasicMap | Map: ...

    @overload
    def __or__(self: Map, other: BasicMap | Map) -> Map: ...

    @overload
    def __or__(self: BasicSet, other: BasicSet | Set) -> BasicSet | Set: ...

    @overload
    def __or__(self: Set, other: BasicSet | Set) -> Set: ...

    def __or__(self, other: _NamedIslSetLike) -> _NamedIslSetLike:
        return _apply_set_like_binary_op(self, other, operator.or_)

    @overload
    def __sub__(self: BasicMap, other: BasicMap | Map) -> BasicMap | Map: ...

    @overload
    def __sub__(self: Map, other: BasicMap | Map) -> Map: ...

    @overload
    def __sub__(self: BasicSet, other: BasicSet | Set) -> BasicSet | Set: ...

    @overload
    def __sub__(self: Set, other: BasicSet | Set) -> Set: ...

    def __sub__(self, other: _NamedIslSetLike) -> _NamedIslSetLike:
        return _apply_set_like_binary_op(self, other, operator.sub)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise TypeError("Objects are not of the same type")

        aligned_self, aligned_other = _align_two(self, other)

        # FIXME: type checker complains because it's not clear whether the
        # underlying object after alignment is an isl.Set
        assert isinstance(aligned_self._obj, isl.Set)
        assert isinstance(aligned_other._obj, isl.Set)
        return aligned_self._obj.plain_is_equal(aligned_other._obj)

    def __lt__(self, other: _NamedIslSetLike) -> bool:
        return _compare_set_like(self, other, isl.Set.is_strict_subset)

    def __le__(self, other: _NamedIslSetLike) -> bool:
        return _compare_set_like(self, other, isl.Set.is_subset)

    def __gt__(self, other: _NamedIslSetLike) -> bool:
        return _compare_set_like(other, self, isl.Set.is_strict_subset)

    def __ge__(self, other: _NamedIslSetLike) -> bool:
        return _compare_set_like(other, self, isl.Set.is_subset)


@final
@dataclass(frozen=True, eq=False)
class BasicSet(_NamedIslSetLike):
    @override
    def add_input_names(self, names_to_add: Collection[str]) -> BasicSet:
        raise NotImplementedError

    @override
    def _reconstruct_isl_object(self) -> isl.BasicSet:
        obj = super()._reconstruct_isl_object()

        if not isinstance(obj, isl.Set) or obj.n_basic_set() != 1:
            raise ValueError(
                "Cannot reconstruct an isl.BasicSet from anything other than "
                "an isl.Set containing only a single isl.BasicSet."
            )

        return obj.get_basic_sets()[0]


@overload
def make_basic_set(src: str, ctx: isl.Context | None = None) -> BasicSet: ...


@overload
def make_basic_set(src: isl.BasicSet) -> BasicSet: ...


def make_basic_set(src: str | isl.BasicSet, ctx: isl.Context | None = None) -> BasicSet:
    obj = isl.BasicSet(src, ctx) if isinstance(src, str) else src
    set_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
    assert isinstance(set_obj, isl.Set)
    return BasicSet(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class Set(_NamedIslSetLike):
    @override
    def add_input_names(self, names_to_add: Collection[str]) -> Set:
        raise NotImplementedError

    @override
    def _reconstruct_isl_object(self) -> isl.Set:
        obj = super()._reconstruct_isl_object()
        assert isinstance(obj, isl.Set)
        return obj

    def get_basic_sets(self) -> Sequence[BasicSet]:
        isl_obj = self._reconstruct_isl_object()

        bsets = isl_obj.get_basic_sets()
        return [make_basic_set(bset) for bset in bsets]


@overload
def _apply_set_like_binary_op(
    lhs: BasicMap, rhs: BasicMap | Map, op: Callable[[isl.Set, isl.Set], isl.Set]
) -> BasicMap | Map: ...


@overload
def _apply_set_like_binary_op(
    lhs: Map, rhs: BasicMap | Map, op: Callable[[isl.Set, isl.Set], isl.Set]
) -> Map: ...


@overload
def _apply_set_like_binary_op(
    lhs: BasicSet, rhs: BasicSet | Set, op: Callable[[isl.Set, isl.Set], isl.Set]
) -> BasicSet | Set: ...


@overload
def _apply_set_like_binary_op(
    lhs: Set, rhs: BasicSet | Set, op: Callable[[isl.Set, isl.Set], isl.Set]
) -> Set: ...


@overload
def _apply_set_like_binary_op(
    lhs: _NamedIslSetLike,
    rhs: _NamedIslSetLike,
    op: Callable[[isl.Set, isl.Set], isl.Set],
) -> _NamedIslSetLike: ...


def _apply_set_like_binary_op(
    lhs: _NamedIslSetLike,
    rhs: _NamedIslSetLike,
    op: Callable[[isl.Set, isl.Set], isl.Set],
) -> _NamedIslSetLike:
    lhs, rhs = _align_two(lhs, rhs)
    result = op(lhs._obj, rhs._obj)

    if isinstance(lhs, BasicMap) and isinstance(rhs, BasicMap):
        result_type = BasicMap if result.n_basic_set() == 1 else Map
    elif isinstance(lhs, BasicSet) and isinstance(rhs, BasicSet):
        result_type = BasicSet if result.n_basic_set() == 1 else Set
    elif isinstance(lhs, (BasicMap, Map)) and isinstance(rhs, (BasicMap, Map)):
        result_type = Map
    else:
        result_type = Set

    return result_type(result, lhs._name_to_dim, lhs._dimtype_to_names)


def _compare_set_like(
    lhs: _NamedIslSetLike, rhs: _NamedIslSetLike, op: Callable[[isl.Set, isl.Set], bool]
) -> bool:
    lhs_is_map = isinstance(lhs, _NamedIslMapLike)
    rhs_is_map = isinstance(rhs, _NamedIslMapLike)
    if lhs_is_map != rhs_is_map:
        raise TypeError("Cannot compare set-like and map-like objects")

    aligned_lhs, aligned_rhs = _align_two(lhs, rhs)

    assert isinstance(aligned_lhs._obj, isl.Set)
    assert isinstance(aligned_rhs._obj, isl.Set)
    return op(aligned_lhs._obj, aligned_rhs._obj)


class _NamedIslMapLike(_NamedIslSetLike):
    @override
    def _reconstruct_isl_object(self) -> isl.BasicMap | isl.Map:
        obj = super()._reconstruct_isl_object()
        if isinstance(obj, isl.Set):
            return isl.Map.from_domain_and_range(isl.Set("{ [] }"), obj)
        assert isinstance(obj, isl.BasicMap | isl.Map)
        return obj

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
    ) -> _NamedIslMapLike:
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
        domain_obj = domain._reconstruct_isl_object()
        assert isinstance(domain_obj, isl.BasicSet | isl.Set)
        result = _apply_set_like_binary_op(
            self,
            self._map_with_universe(
                isl.dim_type.in_,
                domain_obj,
            ),
            operator.and_,
        )
        assert isinstance(result, BasicMap | Map)
        return result

    def intersect_range(self, range_: BasicSet | Set) -> BasicMap | Map:
        range_obj = range_._reconstruct_isl_object()
        assert isinstance(range_obj, isl.BasicSet | isl.Set)
        result = _apply_set_like_binary_op(
            self,
            self._map_with_universe(
                isl.dim_type.out,
                range_obj,
            ),
            operator.and_,
        )
        assert isinstance(result, BasicMap | Map)
        return result

    def apply_range(self, other: BasicMap | Map) -> BasicMap | Map:
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
        return self._wrap_map_result(self._map_obj().reverse())

    def domain(self) -> BasicSet | Set:
        domain = self._map_obj().domain()
        if isinstance(domain, isl.BasicSet):
            return make_basic_set(domain)
        return make_set(domain)

    def range(self) -> BasicSet | Set:
        range_ = self._map_obj().range()
        if isinstance(range_, isl.BasicSet):
            return make_basic_set(range_)
        return make_set(range_)


@overload
def make_set(src: str, ctx: isl.Context | None = None) -> Set: ...


@overload
def make_set(src: isl.Set) -> Set: ...


def make_set(src: isl.Set | str, ctx: isl.Context | None = None) -> Set:
    obj = isl.Set(src, ctx) if isinstance(src, str) else src
    set_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
    assert isinstance(set_obj, isl.Set)
    return Set(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class BasicMap(_NamedIslMapLike):
    @classmethod
    def empty(cls, space: isl.Space) -> BasicMap:
        obj = isl.BasicMap.empty(space)
        set_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
        assert isinstance(set_obj, isl.Set)
        return cls(set_obj, name_to_dim, dimtype_to_names)

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
    obj = isl.BasicMap(src, ctx) if isinstance(src, str) else src
    set_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
    assert isinstance(set_obj, isl.Set)
    return BasicMap(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


def make_map_from_domain_and_range(
    domain: BasicSet | Set, range_: BasicSet | Set
) -> BasicMap | Map:
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
class Map(_NamedIslMapLike):
    @classmethod
    def empty(cls, space: isl.Space) -> Map:
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
    obj = isl.Map(src, ctx) if isinstance(src, str) else src
    set_obj, name_to_dim, dimtype_to_names = _make_named_object_pieces(obj)
    assert isinstance(set_obj, isl.Set)
    return Map(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args
