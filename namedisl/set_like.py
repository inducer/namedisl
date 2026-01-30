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
    _align_and_apply_binary_op,
    _align_two,
    _deconstruct_object,
    _find_contiguous_dim_chunks,
    _strip_names,
)
from .expression_like import PwAff, PwMultiAff, make_pw_aff, make_pw_multi_aff


if TYPE_CHECKING:
    from collections.abc import Sequence


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
        names_to_keep: str | Sequence[str]
    ) -> _NamedIslSetLike:

        if isinstance(names_to_keep, str):
            names_to_keep = [names_to_keep]

        names_to_project_out = [
            name for name in self._name_to_dim
            if name not in names_to_keep
        ]

        return self.project_out(names_to_project_out)

    def dim_max(self, name: str) -> PwAff:
        return make_pw_aff(self._obj.dim_max(self._name_to_dim[name]))

    def dim_min(self, name: str) -> PwAff:
        return make_pw_aff(self._obj.dim_min(self._name_to_dim[name]))

    def as_pw_multi_aff(self) -> PwMultiAff:
        return make_pw_multi_aff(self._reconstruct_isl_object().as_pw_multi_aff())

    # FIXME: basedpyright is not happy with these function signatures
    def __and__(
        self, other: _NamedIslSetLike) -> _NamedIslSetLike:
        return _align_and_apply_binary_op(self, other, operator.and_)

    def __or__(
            self, other: _NamedIslSetLike) -> _NamedIslSetLike:
        return _align_and_apply_binary_op(self, other, operator.or_)

    def __sub__(
            self, other: _NamedIslSetLike) -> _NamedIslSetLike:
        return _align_and_apply_binary_op(self, other, operator.sub)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            raise ValueError("Objects are not of the same type")

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

    return BasicSet(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class Set(_NamedIslSetLike):
    ...


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

    return Set(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class BasicMap(_NamedIslSetLike):

    @override
    def _reconstruct_isl_object(self) -> isl.BasicMap:
        obj = super()._reconstruct_isl_object()

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

    return BasicMap(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class Map(_NamedIslSetLike):
    ...


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

    return Map(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args
