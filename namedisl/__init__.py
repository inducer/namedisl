"""
.. autoclass:: BasicSet

.. autofunction:: make_basic_set
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
from typing import TypeAlias, TypeVar, overload

from constantdict import constantdict

import islpy as isl
from islpy import dim_type


__version__ = metadata.version("namedisl")
_match = re.match(r"^([0-9.]+)([a-z0-9]*?)$", __version__)
assert _match
VERSION = tuple(int(nr) for nr in _match.group(1).split("."))

DIM_TYPES = [dim_type.in_, dim_type.param, dim_type.out]


# {{{ typing

IslObject = TypeVar("IslObject", isl.BasicSet, isl.Set, isl.BasicMap, isl.Map)
NameToDim: TypeAlias = Mapping[str, tuple[isl.dim_type, int]]


@dataclass(frozen=True)
class NamedIslObject:
    _obj: IslObject
    _name_to_dim: NameToDim

    @property
    def space(self):
        return self._obj.space

    def dim_names(self):
        return list(self._name_to_dim.keys())

    def get_dim_id(self, dt, idx):
        return self._obj.get_dim_id(dt, idx)

    def move_dims(self, dim_name, dest_dt, dest_idx, ndims) -> NamedIslObject:
        src_dt, src_idx = self._name_to_dim[dim_name]

        name_to_dim = dict(self._name_to_dim)
        name_to_dim[dim_name] = (dest_dt, dest_idx)
        new_obj = self._obj.move_dims(dest_dt, dest_idx, src_dt, src_idx, ndims)

        return type(self)(new_obj, constantdict(name_to_dim))

    def insert_dims(self, dim_id, dt, idx, ndims) -> NamedIslObject:
        new_obj = self._obj.insert_dims(dt, idx, ndims)
        name_to_dim = dict(self._name_to_dim)
        name_to_dim[dim_id.name] = (dt, idx)

        return type(self)(new_obj, constantdict(name_to_dim))

    def set_dim_id(self, dt, pos, id) -> NamedIslObject:
        new_obj = self._obj.set_dim_id(dt, pos, id)
        return type(self)(new_obj, self._name_to_dim)

    def dim(self, dt) -> int:
        return self._obj.dim(dt)

    def __and__(self, other) -> NamedIslObject:
        return align_and_apply_op(self, other, operator.and_)

    def __getitem__(self, dim_name):
        if dim_name in self._name_to_dim:
            return self._name_to_dim[dim_name]
        else:
            raise ValueError(f"{dim_name} is not a dimension")

    def __str__(self) -> str:
        return str(_restore_names(self._obj, self._name_to_dim))


NamedBinaryOp = Callable[[NamedIslObject, NamedIslObject], NamedIslObject]

# }}}


# {{{ utils

def _strip_names(obj: IslObject) -> tuple[IslObject, NameToDim]:
    name_to_dim = {}
    for tp in isl._CHECK_DIM_TYPES:
        for i in range(obj.dim(tp)):
            name = obj.get_dim_name(tp, i)
            if name is None:
                raise ValueError("unnamed dimension found")
            if name in name_to_dim:
                raise ValueError(f"non-unique dim name: {name}")
            name_to_dim[name] = (tp, i)

            # FIXME: Enable, to avoid misunderstandings
            # obj = obj.set_dim_id(tp, i, None)

    return obj, constantdict(name_to_dim)


def _restore_names(obj: IslObject, name_to_dim: NameToDim) -> IslObject:
    for name, (dt, i) in name_to_dim.items():
        obj = obj.set_dim_name(dt, i, name)

    return obj


def _find_ordering(obj: NamedIslObject, template: NamedIslObject) -> NameToDim:
    """
    Creates a new mapping of dim names to type and index using the union of dim
    names in *template* and *obj*. Dim names sharing a type will be ordered such
    that *template* dim names appear before *obj* dim names.
    """
    name_to_dim = template._name_to_dim
    shared_dim_names = set(obj.dim_names()) & set(template.dim_names())

    dim_indices = {
        dim_type.out: template.dim(dim_type.out),
        dim_type.in_: template.dim(dim_type.in_),
        dim_type.param: template.dim(dim_type.param),
    }
    for name in obj.dim_names():
        if name in shared_dim_names:
            continue

        obj_dt, _ = obj[name]
        name_to_dim = constantdict(
            dict(name_to_dim) | {name: (obj_dt, dim_indices[obj_dt])})

        dim_indices[obj_dt] += 1

    return name_to_dim


def align_spaces(obj: NamedIslObject,
                 template: NamedIslObject,
                 ordering: NameToDim | None = None) -> NamedIslObject:
    """
    Reorders the space of *obj* to match the space of *template*.

    If the set of dimensions in *obj* are not a subset of the dimensions in
    *template*, then a new space is created using the union of the two sets of
    dimensions. The dimensions of *template* are ordered before the dimensions
    of *obj*.
    """
    if ordering is None:
        ordering = _find_ordering(obj, template)

    isl_obj = isl.align_spaces(obj._obj, template._obj, obj_bigger_ok=True)
    isl_obj = _restore_names(isl_obj, ordering)

    return type(obj)(isl_obj, ordering)


def align_two(obj1: NamedIslObject,
              obj2: NamedIslObject) -> Sequence[NamedIslObject]:
    ordering = _find_ordering(obj2, obj1)
    obj2 = align_spaces(obj2, obj1, ordering=ordering)
    obj1 = align_spaces(obj1, obj2, ordering=ordering)

    return obj1, obj2


def align_and_apply_op(
        obj1: NamedIslObject,
        obj2: NamedIslObject,
        op: Callable[[IslObject, IslObject], IslObject]) -> NamedIslObject:
    obj1, obj2 = align_two(obj1, obj2)
    result = op(obj1._obj, obj2._obj)
    return type(obj1)(result, obj1._name_to_dim)

# }}}


# {{{ sets

@dataclass(frozen=True)
class BasicSet(NamedIslObject):
    _obj: isl.BasicSet


@overload
def make_basic_set(src: str, ctx: isl.Context | None = None) -> BasicSet:
    ...


@overload
def make_basic_set(src: isl.BasicSet) -> BasicSet:
    ...


def make_basic_set(src: str | isl.BasicSet,
                   ctx: isl.Context | None = None) -> BasicSet:
    obj = isl.BasicSet(src, ctx) if isinstance(src, str) else src

    obj, name_to_dim = _strip_names(obj)
    return BasicSet(obj, name_to_dim)

# }}}


# {{{ maps

@dataclass(frozen=True)
class BasicMap(NamedIslObject):
    _obj: isl.BasicMap
    _name_to_dim: NameToDim


@overload
def make_basic_map(src: str, ctx: isl.Context | None = None) -> BasicMap:
    ...


@overload
def make_basic_map(src: isl.BasicMap) -> BasicMap:
    ...


def make_basic_map(src: str | isl.BasicMap,
                   ctx: isl.Context | None = None) -> BasicMap:
    obj = isl.BasicMap(src, ctx) if isinstance(src, str) else src

    obj, name_to_dim = _strip_names(obj)
    return BasicMap(obj, name_to_dim)

# }}}
