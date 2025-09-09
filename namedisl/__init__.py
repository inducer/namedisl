"""
.. autoclass:: Set
.. autoclass:: Map

.. autofunction:: make_set
.. autofunction:: make_map
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
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Generic, Self, TypeAlias, TypeVar, overload

from constantdict import constantdict

import islpy as isl
from islpy import dim_type


__version__ = metadata.version("namedisl")
_match = re.match(r"^([0-9.]+)([a-z0-9]*?)$", __version__)
assert _match
VERSION = tuple(int(nr) for nr in _match.group(1).split("."))

ALL_DIM_TYPES = [dim_type.param, dim_type.set, dim_type.in_]


# {{{ typing

class ResultNotCached:
    pass

IslObject = isl.Set | isl.Map
NameToDim: TypeAlias = Mapping[str, tuple[isl.dim_type, int]]

IslTypeT = TypeVar("IslTypeT", bound=IslObject)

# }}}


@dataclass(frozen=True)
class NamedIslObject(Generic[IslTypeT]):
    _obj: IslObject
    _name_to_dim: NameToDim
    # TODO: add cache

    @property
    def dim_names(self):
        return list(self._name_to_dim.keys())

    def add_dim(self, name: str) -> Self:
        ndims = self._obj.dim(dim_type.set)
        obj = self._obj.insert_dims(dim_type.set, ndims, 1)
        name_to_dim = dict(self._name_to_dim) | {name : (dim_type.set, ndims)}

        return type(self)(obj, name_to_dim)


# {{{ utils

def _strip_names(obj: IslObject) -> tuple[IslObject, NameToDim]:
    name_to_dim = {}
    for tp in ALL_DIM_TYPES:
        for i in range(obj.dim(tp)):
            name = obj.get_dim_name(tp, i)
            if name is None:
                raise ValueError("unnamed dimension found")
            if name in name_to_dim:
                raise ValueError(f"non-unique dim name: {name}")
            name_to_dim[name] = (tp, i)

    return obj, constantdict(name_to_dim)


def _restore_names(named_obj: NamedIslObject) -> IslObject:
    obj = named_obj._obj
    name_to_dim = named_obj._name_to_dim

    for name, (dt, i) in name_to_dim.items():
        obj = obj.set_dim_name(dt, i, name)

    return obj


def _find_joint_name_to_dim(obj: NameToDim, template: NameToDim) -> NameToDim:
    """
    Uses `template` to determine a name-to-dimension mapping used for aligning
    the spaces between `obj` and `template`.
    """
    name_to_dim = dict(template)

    shared_names = set(name_to_dim.keys()) & set(obj.keys())
    for name in sorted(shared_names):
        dim_type_obj, _ = obj[name] 
        dim_type_template, _ = template[name]
        if dim_type_obj != dim_type_template:
            raise ValueError(
                f"{name} belongs to a different dim_type in `obj` than in "
                "`template`"
            )

    dim_type_to_idx = dict.fromkeys(ALL_DIM_TYPES, -1) 
    for dim_type, pos in template.values():
        dim_type_to_idx[dim_type] = max(dim_type_to_idx[dim_type], pos + 1)
     
    unique_names = set(obj.keys()) - shared_names
    for name in sorted(unique_names):
        dim_type, _ = obj[name]
        pos = dim_type_to_idx[dim_type]
        name_to_dim = constantdict(dict(name_to_dim) | {name: (dim_type, pos)})

        dim_type_to_idx[dim_type] += 1

    return name_to_dim


def _align_space(obj: NamedIslObject, ordering: NameToDim) -> NamedIslObject:
    """
    Aligns the space and name-to-dimension mapping of `obj` to match what is
    specified by `ordering`. Returns a new object whose dims are aligned
    according to `ordering`.
    """
    new_isl_obj = obj._obj.copy()
    new_name_to_dim = dict(obj._name_to_dim)

    for name, (dim_type, pos) in sorted(ordering.items(), key=lambda k: k[1][1]):
        if name in obj._name_to_dim:
            _, old_pos = new_name_to_dim[name]
            if old_pos == pos: 
                    continue

            temp_dim_type = isl.dim_type.param
            if temp_dim_type == dim_type:
                temp_dim_type = isl.dim_type.set

            temp_pos = new_isl_obj.dim(temp_dim_type)

            new_isl_obj = new_isl_obj.move_dims(
                temp_dim_type, temp_pos, dim_type, old_pos, 1)
            new_isl_obj = new_isl_obj.move_dims(
                dim_type, pos, temp_dim_type, temp_pos, 1)
        else:
            old_pos = new_isl_obj.dim(dim_type)
            new_isl_obj = new_isl_obj.insert_dims(dim_type, pos, 1)

            if dim_type == isl.dim_type.param:
                new_isl_obj = new_isl_obj.set_dim_name(dim_type, pos, name)

        temp_name_to_dim = new_name_to_dim.copy()
        for cur_name, (cur_dt, cur_pos) in sorted(new_name_to_dim.items(),
                                                  key=lambda k: k[1][1]):
            if cur_dt != dim_type:
                continue

            if (pos > old_pos) and (cur_pos > old_pos):
                temp_name_to_dim[cur_name] = (cur_dt, cur_pos - 1)
            elif (pos < old_pos) and (cur_pos < old_pos):
                temp_name_to_dim[cur_name] = (cur_dt, cur_pos + 1)
            
        new_name_to_dim = temp_name_to_dim
        new_name_to_dim[name] = (dim_type, pos)

    return type(obj)(new_isl_obj, ordering)


def _align_two(obj1: NamedIslObject,
               obj2: NamedIslObject) -> Sequence[NamedIslObject]:
    """
    Aligns the spaces and name-to-dimension mappings of `obj1` and `obj2` so
    that they are compatible for *named* set operations. `obj2` will first be
    aligned to `obj1`, then `obj1` will be aligned to the result of the first
    alignment.
    """
    ordering = _find_joint_name_to_dim(obj2._name_to_dim, obj1._name_to_dim)

    obj2 = _align_space(obj2, ordering)
    obj1 = _align_space(obj1, ordering)

    return obj1, obj2


def _align_and_apply_op(
        obj1: NamedIslObject,
        obj2: NamedIslObject,
        op: Callable[[IslObject, IslObject], IslObject]) -> NamedIslObject:

    obj1, obj2 = _align_two(obj1, obj2)

    result = op(obj1._obj, obj2._obj)

    return type(obj1)(result, obj1._name_to_dim)

# }}}

# {{{ Set-like objects

@dataclass(frozen=True)
class _SetLike(NamedIslObject):
    def complement(self) -> Self:
        return type(self)(self._obj.complement(), self._name_to_dim)

    def project_out(self, names_to_project_out: str | Sequence[str]) -> Self:

        if isinstance(names_to_project_out, str):
            names_to_project_out = [names_to_project_out]

        new_name_to_dim = dict(self._name_to_dim)
        new_isl_obj = self._obj
        for proj_name in names_to_project_out:
            dt, pos = new_name_to_dim[proj_name]
            new_name_to_dim.pop(proj_name)

            temp_name_to_dim = new_name_to_dim.copy()
            for name, (cur_dt, cur_pos) in sorted(
                    new_name_to_dim.items(), key=lambda k: k[1][1]):
                if cur_pos > pos:
                    temp_name_to_dim[name] = (cur_dt, cur_pos - 1)

            new_name_to_dim = temp_name_to_dim.copy()

            new_isl_obj = new_isl_obj.project_out(dt, pos, 1)

        return type(self)(new_isl_obj, new_name_to_dim)

    def eliminate(self, names: str | Sequence[str]) -> Self:

        if isinstance(names, str):
            names = [names]

        new_obj = self._obj
        for name in names:
            dt, pos = self._name_to_dim[name]

            new_obj = new_obj.eliminate(dt, pos, 1)

        return type(self)(new_obj, self._name_to_dim)

    def __and__(self, other) -> Self:
        return _align_and_apply_op(self, other, operator.and_)

    def __eq__(self, other) -> bool:
        aligned_self, aligned_other = _align_two(self, other)
        return aligned_self._obj.plain_is_equal(aligned_other._obj)

    def __or__(self, other) -> Self:
        return _align_and_apply_op(self, other, operator.or_)

    def __sub__(self, other) -> Self:
        return _align_and_apply_op(self, other, operator.sub)

    def __str__(self) -> str:
        return str(_restore_names(self))


@dataclass(frozen=True, eq=False)
class Set(_SetLike):
    _obj: isl.Set


@overload
def make_set(src: str, ctx: isl.Context | None = None) -> Set:
    ...


@overload
def make_set(src: isl.Set) -> Set:
    ...


def make_set(src: str | isl.Set, ctx: isl.Context | None = None) -> Set:
    obj = isl.Set(src, ctx) if isinstance(src, str) else src

    obj, name_to_dim = _strip_names(obj)
    return Set(obj, name_to_dim)


@dataclass(frozen=True, eq=False)
class Map(_SetLike):
    _obj: isl.Map


@overload
def make_map(src: str, ctx: isl.Context | None = None) -> Map:
    ...


@overload
def make_map(src: isl.Map) -> Map:
    ...


def make_map(src: str | isl.Map,
             ctx: isl.Context | None = None) -> Map:
    obj = isl.Map(src, ctx) if isinstance(src, str) else src

    obj, name_to_dim = _strip_names(obj)
    return Map(obj, name_to_dim)

# }}}
