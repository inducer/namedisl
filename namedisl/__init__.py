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
from typing import TypeAlias, TypeVar, overload

from constantdict import constantdict

import islpy as isl


__version__ = metadata.version("namedisl")
_match = re.match(r"^([0-9.]+)([a-z0-9]*?)$", __version__)
assert _match
VERSION = tuple(int(nr) for nr in _match.group(1).split("."))

IslObject = TypeVar("IslObject", isl.BasicSet, isl.Set, isl.BasicMap, isl.Map)
NameToDim: TypeAlias = Mapping[str, tuple[isl.dim_type, int]]


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

# {{{ sets

@dataclass(frozen=True)
class BasicSet:
    _obj: isl.BasicSet
    _name_to_dim: NameToDim

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
