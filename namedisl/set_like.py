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
from abc import ABC
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, final, overload

from constantdict import constantdict
from typing_extensions import override

import islpy as isl

from .core import (
    NamedIslObject,
    NameToDim,
    _align_two,
    _deconstruct_object,
    _find_contiguous_dim_chunks,
    _restore_names,
    _strip_names,
)


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

    # {{{ TODO: functions that return ExpressionLike objects

    def dim_max(self, name: str):
        ...

    def dim_min(self, name: str):
        ...

    def as_pw_multi_aff(self):
        ...

    # }}}


@final
@dataclass(frozen=True, eq=False)
class Set(_NamedIslSetLike):
    @override
    def _reconstruct_isl_object(self) -> isl.Set:
        # FIXME: typechecker complains that self._obj is not an isl.Set even
        # though _NamedIslObject is instantiated with isl.Set.
        # using reveal_type(self._obj) below shows self._obj is
        # isl.Set | isl.Map?
        assert isinstance(self._obj, isl.Set)

        if self._has_params:
            if self._parameter_dim_start is None:
                raise ValueError(
                    "Object has parameter dimensions, but a starting index for "
                    "parameter names is not given. Reconstruction is not "
                    "possible")

            return self._obj.move_dims(
                isl.dim_type.param, 0,
                isl.dim_type.set, self._parameter_dim_start,
                    len(self._parameter_names)
            )

        return self._obj

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Set):
            raise NotImplementedError

        aligned_self, aligned_other = _align_two(self, other)

        # FIXME: type checker complains because it's not clear whether the
        # underlying object after alignment is an isl.Set
        assert isinstance(aligned_other._obj, isl.Set)
        assert isinstance(aligned_self._obj, isl.Set)
        return aligned_self._obj.plain_is_equal(aligned_other._obj)


@overload
def make_set(src: str, ctx: isl.Context | None = None) -> Set:
    ...


@overload
def make_set(src: isl.Set) -> Set:
    ...


def make_set(src: isl.Set | str, ctx: isl.Context | None = None) -> Set:
    obj = isl.Set(src, ctx) if isinstance(src, str) else src

    set_obj, dimtype_to_names = _deconstruct_object(obj)
    set_obj, name_to_dim = _strip_names(set_obj)

    assert isinstance(set_obj, isl.Set)
    return Set(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args


@final
@dataclass(frozen=True, eq=False)
class Map(_NamedIslSetLike):
    @override
    def _reconstruct_isl_object(self) -> isl.Map:
        """
        Relies on the dimension type ordering in
        :func:`_deconstruct_set_like_object`.
        """
        if self._input_dim_start is None:
            raise ValueError("Cannot reconstruct a map object without knowledge "
                             "of the starting position of input dimensions")

        obj = _restore_names(self._obj, self._name_to_dim)
        assert isinstance(obj, isl.Set)

        obj_domain = isl.Set("{ [] }")
        obj_range = obj

        map_obj = isl.Map.from_domain_and_range(obj_domain, obj_range)

        if self._has_params:
            if self._parameter_dim_start is None:
                raise ValueError(
                    "Object has parameter dimensions, but a starting index for "
                    "parameter names is not given. Reconstruction is not "
                    "possible")

            param_start = self._parameter_dim_start
            map_obj = map_obj.move_dims(
                isl.dim_type.param, 0,
                isl.dim_type.set, param_start, len(self._parameter_names)
            )

        inp_start = self._input_dim_start
        map_obj = map_obj.move_dims(
            isl.dim_type.in_, 0,
            isl.dim_type.set, inp_start, len(self._input_names)
        )

        return map_obj

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Map):
            raise NotImplementedError

        aligned_self, aligned_other = _align_two(self, other)

        # FIXME: type checker complains because it's not clear whether the
        # underlying object after alignment is an isl.Set
        assert isinstance(aligned_self._obj, isl.Set)
        assert isinstance(aligned_other._obj, isl.Set)
        return aligned_self._obj.plain_is_equal(aligned_other._obj)


@overload
def make_map(src: str, ctx: isl.Context | None = None) -> Map:
    ...


@overload
def make_map(src: isl.Map) -> Map:
    ...


def make_map(src: str | isl.Map, ctx: isl.Context | None = None) -> Map:
    obj = isl.Map(src, ctx) if isinstance(src, str) else src

    set_obj, dimtype_to_names = _deconstruct_object(obj)
    set_obj, name_to_dim = _strip_names(set_obj)

    assert isinstance(set_obj, isl.Set)
    return Map(set_obj, name_to_dim, dimtype_to_names)  # pylint: disable=too-many-function-args
