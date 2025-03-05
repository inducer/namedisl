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

import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Sequence, TypeAlias, TypeVar, overload

import operator

from constantdict import constantdict

import islpy as isl


__version__ = metadata.version("namedisl")
_match = re.match(r"^([0-9.]+)([a-z0-9]*?)$", __version__)
assert _match
VERSION = tuple(int(nr) for nr in _match.group(1).split("."))


# {{{ typing

IslObject = TypeVar("IslObject", isl.BasicSet, isl.Set, isl.BasicMap, isl.Map)
NameToDim: TypeAlias = Mapping[str, tuple[isl.dim_type, int]]

@dataclass(frozen=True)
class NamedIslObject:
    _obj: IslObject
    _name_to_dim: NameToDim

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


def _find_alignment_ordering(obj: NamedIslObject,
                             template: NamedIslObject) -> Sequence:
    return list(obj._name_to_dim.keys())


def _apply_ordering(obj: NamedIslObject,
                    ordering: Sequence) -> NamedIslObject:
    return obj


def align_spaces(obj: NamedIslObject,
                 template: NamedIslObject,
                 obj_bigger_ok: bool = False) -> NamedIslObject:
    return _apply_ordering(obj, _find_alignment_ordering(obj, template))


def align_two(obj1: NamedIslObject,
              obj2: NamedIslObject) -> Sequence[NamedIslObject]:
    obj1 = align_spaces(obj1, obj2, obj_bigger_ok=True)
    obj2 = align_spaces(obj2, obj1, obj_bigger_ok=True)
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

    def __and__(self, other) -> BasicSet:
        return align_and_apply_op(self, other, operator.and_)

    def __str__(self) -> str:
        return str(_restore_names(self._obj, self._name_to_dim))


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
class BasicMap:
    _obj: isl.BasicMap
    _name_to_dim: NameToDim

    def __and__(self, other) -> BasicMap:
        if self._name_to_dim == other._name_to_dim:
            return BasicMap(self._obj & other._obj, self._name_to_dim)
        else: # do some analysis to figure out whether intersection is legal
            pass
        raise ValueError("Name mismatch, cannot intersect maps")

    def __str__(self) -> str:
        return str(_restore_names(self._obj, self._name_to_dim))


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
