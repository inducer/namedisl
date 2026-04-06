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
from typing_extensions import override

import islpy as isl

from .core import (
    NamedIslObject,
    NameToDim,
    _align_obj,
    _align_two,
    _deconstruct_object,
    _find_contiguous_dim_chunks,
    _normalize_dimtype_to_names,
    _strip_names,
)
from .expression_like import PwAff, PwMultiAff, make_pw_aff, make_pw_multi_aff


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True, eq=False)
class _NamedIslSetLike(NamedIslObject[isl.Set], ABC):
    """
    Represents set-like objects with parameter dimensions as a non-parameterized
    set. Names are organized as contiguous chunks of dimension types, i.e.
        [ (set names), (input names), (parameter names) ]
    """

    def complement(self: _NamedIslSetLike) -> _NamedIslSetLike:
        return replace(
            self,
            _obj=self._obj.complement(),
            _name_to_dim=self._name_to_dim,
            _dimtype_to_names=self._dimtype_to_names
        )

    def eliminate(self, names_to_eliminate: str | Sequence[str]) -> _NamedIslSetLike:
        if isinstance(names_to_eliminate, str):
            names_to_eliminate = [names_to_eliminate]

        dims_to_eliminate = sorted(
            self._name_to_dim[name]
            for name in names_to_eliminate
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
            _dimtype_to_names=self._dimtype_to_names
        )

    def project_out(self: _NamedIslSetLike,
                    names_to_project_out: str | Sequence[str]) -> _NamedIslSetLike:

        if isinstance(names_to_project_out, str):
            names_to_project_out = [names_to_project_out]

        names_to_remove = set(names_to_project_out)

        dims_to_remove = sorted(
            self._name_to_dim[name]
            for name in names_to_remove
        )

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
            _dimtype_to_names=new_type_to_names
        )

    def project_out_except(
        self: _NamedIslSetLike,
        names_to_keep: str | Sequence[str],
        dim_types: Sequence[isl.dim_type] | None = None
    ) -> _NamedIslSetLike:

        if isinstance(names_to_keep, str):
            names_to_keep = [names_to_keep] if names_to_keep else []

        considered_names = set(self._name_to_dim)
        if dim_types is not None:
            considered_names = set()
            for dim_type in dim_types:
                if dim_type == isl.dim_type.param:
                    considered_names |= set(self.parameter_names)
                elif dim_type in (
                        isl.dim_type.set,
                        isl.dim_type.out,
                        isl.dim_type.in_
                    ):
                    considered_names |= (
                        set(self._name_to_dim)
                        - set(self.parameter_names)
                        - set(self.input_names)
                    )

        names_to_project_out = [
            name for name in considered_names
            if name not in names_to_keep
        ]

        return self.project_out(names_to_project_out)

    def dim_max(self, name: str | int) -> PwAff:
        dim = name if isinstance(name, int) else self._name_to_dim[name]
        return make_pw_aff(self._obj.dim_max(dim))

    def dim_min(self, name: str | int) -> PwAff:
        dim = name if isinstance(name, int) else self._name_to_dim[name]
        return make_pw_aff(self._obj.dim_min(dim))

    def as_pw_multi_aff(self) -> PwMultiAff:
        return make_pw_multi_aff(self._reconstruct_isl_object().as_pw_multi_aff())

    @override
    def dim(self, dim_type: isl.dim_type) -> int:
        if dim_type == isl.dim_type.out:
            dim_type = isl.dim_type.set
        return self._reconstruct_isl_object().dim(dim_type)

    @override
    def get_dim_name(self, dim_type: isl.dim_type, dim: int) -> str | None:
        if dim_type == isl.dim_type.out:
            dim_type = isl.dim_type.set
        return self._reconstruct_isl_object().get_dim_name(dim_type, dim)

    # FIXME: basedpyright is not happy with these function signatures
    def __and__(
        self, other: _NamedIslSetLike) -> _NamedIslSetLike:
        return _apply_set_like_binary_op(self, other, operator.and_)

    def __or__(
            self, other: _NamedIslSetLike) -> _NamedIslSetLike:
        return _apply_set_like_binary_op(self, other, operator.or_)

    def __sub__(
            self, other: _NamedIslSetLike) -> _NamedIslSetLike:
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


@final
@dataclass(frozen=True, eq=False)
class BasicSet(_NamedIslSetLike):
    @override
    def add_input_names(self, names_to_add: Sequence[str]) -> BasicSet:
        raise NotImplementedError

    @override
    def add_output_names(self, names_to_add: Sequence[str]) -> BasicSet:
        raise NotImplementedError

    @override
    def _reconstruct_isl_object(self) -> isl.BasicSet:
        obj = super()._reconstruct_isl_object()

        if not isinstance(obj, isl.Set) or obj.n_basic_set() != 1:
            raise ValueError(
                "Cannot reconstruct an isl.BasicSet from anything other than "
                "an isl.Set containing only a single isl.BasicSet.")

        return obj.get_basic_sets()[0]


@overload
def make_basic_set(src: str, ctx: isl.Context | None = None) -> BasicSet:
    ...


@overload
def make_basic_set(src: isl.BasicSet) -> BasicSet:
    ...


def make_basic_set(src: str | isl.BasicSet, ctx: isl.Context | None = None) -> BasicSet:
    obj = isl.BasicSet(src, ctx) if isinstance(src, str) else src

    set_obj, dimtype_to_names = _deconstruct_object(obj)

    assert isinstance(set_obj, isl.Set)
    set_obj, name_to_dim = _strip_names(set_obj)
    dimtype_to_names = _normalize_dimtype_to_names(set_obj, dimtype_to_names)

    return BasicSet(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class Set(_NamedIslSetLike):
    ...


def _apply_set_like_binary_op(
        lhs: _NamedIslSetLike,
        rhs: _NamedIslSetLike,
        op: Callable[[isl.Set, isl.Set], isl.Set]
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


class _NamedIslMapLike(_NamedIslSetLike):
    @override
    def _reconstruct_isl_object(self) -> isl.Map:
        obj = super()._reconstruct_isl_object()
        if isinstance(obj, isl.Set):
            return isl.Map.from_domain_and_range(isl.Set("{ [] }"), obj)
        return obj

    def _output_names(self) -> frozenset[str]:
        return frozenset(self._name_to_dim) - self.input_names - self.parameter_names

    @staticmethod
    def _logical_name(name: str) -> str:
        return name.rstrip("'")

    def _ordered_names(self, names: frozenset[str]) -> tuple[str, ...]:
        return tuple(sorted(names, key=self._name_to_dim.__getitem__))

    def _ordered_logical_names(self, names: frozenset[str]) -> tuple[str, ...]:
        return tuple(self._logical_name(name) for name in self._ordered_names(names))

    def _actual_names_for_logical_order(
            self,
            names: frozenset[str],
            logical_order: tuple[str, ...]
        ) -> tuple[str, ...]:
        name_by_logical: dict[str, str] = {}
        for name in self._ordered_names(names):
            logical_name = self._logical_name(name)
            if logical_name in name_by_logical:
                raise ValueError(
                    "multiple dimensions in one interface share the same "
                    f"logical name: {logical_name}"
                )
            name_by_logical[logical_name] = name

        try:
            return tuple(name_by_logical[logical_name] for logical_name in logical_order)
        except KeyError as exc:
            raise ValueError("maps are not composable: interface names differ") from exc

    def _reorder_interface(
            self,
            dim_type: isl.dim_type,
            logical_order: tuple[str, ...]
        ) -> _NamedIslMapLike:
        interface_names = (
            self.input_names if dim_type == isl.dim_type.in_ else self._output_names()
        )
        ordered_names = self._actual_names_for_logical_order(
            interface_names,
            logical_order
        )
        current_names = self._ordered_names(interface_names)
        if current_names == ordered_names:
            return self

        out_names = (
            ordered_names if dim_type == isl.dim_type.out
            else self._ordered_names(self._output_names())
        )
        in_names = (
            ordered_names if dim_type == isl.dim_type.in_
            else self._ordered_names(self.input_names)
        )
        param_names = self._ordered_names(self.parameter_names)

        ordering: NameToDim = constantdict({
            name: dim
            for dim, name in enumerate((*out_names, *in_names, *param_names))
        })

        return _align_obj(self, ordering, self._dimtype_to_names)

    def _validate_composable(
            self,
            lhs_dim_type: isl.dim_type,
            other: BasicMap | Map,
            rhs_dim_type: isl.dim_type
        ) -> tuple[str, ...]:
        lhs_names = self.input_names if lhs_dim_type == isl.dim_type.in_ else self._output_names()
        rhs_names = other.input_names if rhs_dim_type == isl.dim_type.in_ else other._output_names()
        if (
                frozenset(self._logical_name(name) for name in lhs_names)
                !=
                frozenset(self._logical_name(name) for name in rhs_names)
            ):
            raise ValueError("maps are not composable: interface names differ")
        return self._ordered_logical_names(lhs_names)

    def intersect_domain(self, domain: BasicSet | Set) -> Map:
        return self & make_map(isl.Map.from_domain_and_range(
            domain._reconstruct_isl_object(),
            isl.Set.universe(self._reconstruct_isl_object().range().get_space())
        ))

    def intersect_range(self, range_: BasicSet | Set) -> Map:
        return self & make_map(isl.Map.from_domain_and_range(
            isl.Set.universe(self._reconstruct_isl_object().domain().get_space()),
            range_._reconstruct_isl_object()
        ))

    def apply_range(self, other: BasicMap | Map) -> Map:
        ordered_names = self._validate_composable(isl.dim_type.out, other, isl.dim_type.in_)
        other = other._reorder_interface(isl.dim_type.in_, ordered_names)
        return make_map(
            self._reconstruct_isl_object().apply_range(other._reconstruct_isl_object())
        )

    def apply_domain(self, other: BasicMap | Map) -> Map:
        ordered_names = self._validate_composable(isl.dim_type.in_, other, isl.dim_type.out)
        other = other._reorder_interface(isl.dim_type.out, ordered_names)
        return make_map(
            other._reconstruct_isl_object().apply_range(self._reconstruct_isl_object())
        )

    def reverse(self) -> Map:
        return make_map(self._reconstruct_isl_object().reverse())

    def domain(self) -> Set:
        return make_set(self._reconstruct_isl_object().domain())

    def range(self) -> Set:
        return make_set(self._reconstruct_isl_object().range())


@overload
def make_set(src: str, ctx: isl.Context | None = None) -> Set:
    ...


@overload
def make_set(src: isl.Set) -> Set:
    ...


def make_set(src: isl.Set | str, ctx: isl.Context | None = None) -> Set:
    obj = isl.Set(src, ctx) if isinstance(src, str) else src

    set_obj, dimtype_to_names = _deconstruct_object(obj)

    assert isinstance(set_obj, isl.Set)
    set_obj, name_to_dim = _strip_names(set_obj)
    dimtype_to_names = _normalize_dimtype_to_names(set_obj, dimtype_to_names)

    return Set(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class BasicMap(_NamedIslMapLike):
    @classmethod
    def empty(cls, space: isl.Space) -> BasicMap:
        obj = isl.BasicMap.empty(space)
        set_obj, dimtype_to_names = _deconstruct_object(obj)
        assert isinstance(set_obj, isl.Set)
        set_obj, name_to_dim = _strip_names(set_obj)
        dimtype_to_names = _normalize_dimtype_to_names(set_obj, dimtype_to_names)
        return cls(set_obj, name_to_dim, dimtype_to_names)

    def reverse(self) -> BasicMap:
        return make_basic_map(self._reconstruct_isl_object().reverse())

    def domain(self) -> BasicSet:
        return make_basic_set(self._reconstruct_isl_object().domain())

    def range(self) -> BasicSet:
        return make_basic_set(self._reconstruct_isl_object().range())

    def intersect_domain(self, domain: BasicSet | Set) -> BasicMap | Map:
        if isinstance(domain, BasicSet):
            return self & make_basic_map(isl.BasicMap.from_domain_and_range(
                domain._reconstruct_isl_object(),
                isl.BasicSet.universe(self._reconstruct_isl_object().range().get_space())
            ))
        return super().intersect_domain(domain)

    def intersect_range(self, range_: BasicSet | Set) -> BasicMap | Map:
        if isinstance(range_, BasicSet):
            return self & make_basic_map(isl.BasicMap.from_domain_and_range(
                isl.BasicSet.universe(self._reconstruct_isl_object().domain().get_space()),
                range_._reconstruct_isl_object()
            ))
        return super().intersect_range(range_)

    def apply_range(self, other: BasicMap | Map) -> BasicMap | Map:
        if isinstance(other, BasicMap):
            ordered_names = self._validate_composable(
                isl.dim_type.out, other, isl.dim_type.in_)
            other = other._reorder_interface(isl.dim_type.in_, ordered_names)
            return make_basic_map(
                self._reconstruct_isl_object().apply_range(
                    other._reconstruct_isl_object())
            )
        return super().apply_range(other)

    def apply_domain(self, other: BasicMap | Map) -> BasicMap | Map:
        if isinstance(other, BasicMap):
            ordered_names = self._validate_composable(
                isl.dim_type.in_, other, isl.dim_type.out)
            other = other._reorder_interface(isl.dim_type.out, ordered_names)
            return make_basic_map(
                other._reconstruct_isl_object().apply_range(
                    self._reconstruct_isl_object())
            )
        return super().apply_domain(other)

    @override
    def _reconstruct_isl_object(self) -> isl.BasicMap:
        obj = super()._reconstruct_isl_object()

        if isinstance(obj, isl.Map) and obj.is_empty():
            return isl.BasicMap.empty(obj.get_space())

        if not isinstance(obj, isl.Map) or obj.n_basic_map() != 1:
            raise ValueError(
                "Cannot reconstruct an isl.BasicMap from anything other than "
                "an isl.Map containing only a single isl.BasicMap.")

        return obj.get_basic_maps()[0]


@overload
def make_basic_map(src: str, ctx: isl.Context | None = None) -> BasicMap:
    ...


@overload
def make_basic_map(src: isl.BasicMap) -> BasicMap:
    ...


def make_basic_map(src: str | isl.BasicMap, ctx: isl.Context | None = None) -> BasicMap:
    obj = isl.BasicMap(src, ctx) if isinstance(src, str) else src

    set_obj, dimtype_to_names = _deconstruct_object(obj)

    assert isinstance(set_obj, isl.Set)
    set_obj, name_to_dim = _strip_names(set_obj)
    dimtype_to_names = _normalize_dimtype_to_names(set_obj, dimtype_to_names)

    return BasicMap(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


def make_map_from_domain_and_range(
        domain: BasicSet | Set,
        range_: BasicSet | Set
    ) -> BasicMap | Map:
    if isinstance(domain, BasicSet) and isinstance(range_, BasicSet):
        return make_basic_map(
            isl.BasicMap.from_domain_and_range(
                domain._reconstruct_isl_object(),
                range_._reconstruct_isl_object()
            )
        )

    return make_map(
        isl.Map.from_domain_and_range(
            domain._reconstruct_isl_object(),
            range_._reconstruct_isl_object()
        )
    )


@final
@dataclass(frozen=True, eq=False)
class Map(_NamedIslMapLike):
    @classmethod
    def empty(cls, space: isl.Space) -> Map:
        return make_map(isl.Map.empty(space))


@overload
def make_map(src: str, ctx: isl.Context | None = None) -> Map:
    ...


@overload
def make_map(src: isl.Map) -> Map:
    ...


def make_map(src: str | isl.Map, ctx: isl.Context | None = None) -> Map:
    obj = isl.Map(src, ctx) if isinstance(src, str) else src

    set_obj, dimtype_to_names = _deconstruct_object(obj)

    assert isinstance(set_obj, isl.Set)
    set_obj, name_to_dim = _strip_names(set_obj)
    dimtype_to_names = _normalize_dimtype_to_names(set_obj, dimtype_to_names)

    return Map(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args
