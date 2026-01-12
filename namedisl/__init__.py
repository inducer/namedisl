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
from dataclasses import dataclass
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from importlib import metadata
from typing import Generic, TypeAlias, TypeVar, final, overload

from constantdict import constantdict
from typing_extensions import override

import islpy as isl


__version__ = metadata.version("namedisl")
_match = re.match(r"^([0-9.]+)([a-z0-9]*?)$", __version__)
assert _match
VERSION = tuple(int(nr) for nr in _match.group(1).split("."))


IslExpressionLikeT = TypeVar("IslExpressionLikeT", isl.Aff, isl.QPolynomial)
IslSetLikeT = TypeVar("IslSetLikeT", isl.Set, isl.Map)
IslObjectT = TypeVar("IslObjectT", isl.Set, isl.Map)

IslSetLike = isl.Set | isl.Map
IslExpressionLike = isl.Aff | isl.QPolynomial

SetLikePieces: TypeAlias = tuple[isl.Set, tuple[frozenset[str], ...]]

NameToDim: TypeAlias = Mapping[str, int]


def _strip_names(obj: IslObjectT) -> tuple[IslObjectT, NameToDim]:
    name_to_dim: dict[str, int] = {}
    for i in range(obj.dim(isl.dim_type.set)):
        name = obj.get_dim_name(isl.dim_type.set, i)

        if name is None:
            raise ValueError("unnamed dimension found")

        if name in name_to_dim:
            raise ValueError(f"non-unique dim name: {name}")

        name_to_dim[name] = i

    return obj, constantdict(name_to_dim)


def _get_dim_names(obj: IslObjectT, dt: isl.dim_type) -> frozenset[str]:
    all_dt_names: list[str] = []
    for dim in range(obj.dim(dt)):
        name = obj.get_dim_name(dt, dim)

        if name is None:
            raise ValueError("unnamed dimension found")

        all_dt_names.append(name)

    return frozenset(all_dt_names)


def _deconstruct_set_like_object(obj: IslSetLikeT) -> SetLikePieces:
    from islpy import dim_type

    dt_to_names: dict[dim_type, frozenset[str]] = {}
    for dt in dt_to_names.keys():
        dt_to_names[dt] = _get_dim_names(obj, dt)
        obj = obj.move_dims(
            dim_type.set,
            obj.dim(dim_type.set),
            dt,
            0,
            obj.dim(dt)
        )

    if isinstance(obj, isl.Map):
        set_obj = obj.range()
    else:
        set_obj = obj

    input_names = dt_to_names[dim_type.in_]
    param_names = dt_to_names[dim_type.param]

    if input_names:
        input_names = frozenset(input_names)
    else:
        input_names = frozenset()

    if param_names:
        param_names = frozenset(param_names)
    else:
        param_names = frozenset()

    return set_obj, (input_names, param_names)


@dataclass(frozen=True)
class NamedIslObject(Generic[IslObjectT], ABC):
    _obj: IslObjectT
    _name_to_dim: NameToDim

    _parameter_names: frozenset[str]
    _parameter_dim_start: int

    # NOTE: defaulting these for all subclasses reduces the amount of
    # specialization when aligning spaces of objects
    _input_names: frozenset[str] = frozenset()
    _input_dim_start: int = -1

    @abstractmethod
    def _reconstruct_isl_object(self) -> IslExpressionLike | IslSetLike:
        ...

    @override
    def __str__(self) -> str:
        return str(self._reconstruct_isl_object())


# FIXME: enforcing alphabetical ordering within each contiguous chunk of
# dimension types solves the problem
def _find_joint_name_to_dim(
        obj: NamedIslObject[IslObjectT],
        other: NamedIslObject[IslObjectT]
    ) -> tuple[NameToDim, tuple[frozenset[str], frozenset[str]]]:
    """
    Constructs a mapping from names to dimensions such that names within each
    "type chunk" are sorted alphabetically. Specifically, the internal
    :class:`isl.Set` representation of each :class:`NamedIslObject` will have
    the form

    [ (set dimensions), (parameter dimensions), (input_dimensions) ]

    where the names in each dimension appear in alphabetical order.
    """
    obj_all_names = frozenset(obj._name_to_dim.keys())
    obj_inp_names = obj._input_names
    obj_param_names = obj._parameter_names
    obj_set_names = (obj_all_names - obj_param_names) - obj_inp_names

    other_all_names = frozenset(other._name_to_dim.keys())
    other_inp_names = other._input_names
    other_param_names = other._parameter_names
    other_set_names = (other_all_names - other_param_names) - other_inp_names

    all_inp_names = sorted(list(obj_inp_names | other_inp_names))
    all_param_names = sorted(list(obj_param_names | other_param_names))
    all_set_names = sorted(list(obj_set_names | other_set_names))
    all_names = all_set_names + all_param_names + all_inp_names

    name_to_dim = { name : dim for dim, name in enumerate(all_names) }

    return name_to_dim, (frozenset(all_param_names), frozenset(all_inp_names))


@dataclass(frozen=True)
class _NamedIslSetLike(NamedIslObject[isl.Set], ABC):
    _obj: isl.Set


@final
@dataclass(frozen=True, eq=False)
class Set(_NamedIslSetLike):
    @override
    def _reconstruct_isl_object(self) -> isl.Set:
        return self._obj.move_dims(
            isl.dim_type.param, 0,
            isl.dim_type.set, self._parameter_dim_start,
                len(self._parameter_names)
        )


@overload
def make_set(src: str, ctx: isl.Context | None = None) -> Set:
    ...


@overload
def make_set(src: isl.Set) -> Set:
    ...


def make_set(src: isl.Set | str, ctx: isl.Context | None = None) -> Set:
    obj = isl.Set(src, ctx) if isinstance(src, str) else src

    set_obj, (param_names, _) = _deconstruct_set_like_object(obj)
    set_obj, name_to_dim = _strip_names(set_obj)
    parameter_dim_start = min(
        name_to_dim[name]
        for name in param_names
    )

    return Set(set_obj, name_to_dim, param_names, parameter_dim_start)


@final
@dataclass(frozen=True, eq=False)
class Map(_NamedIslSetLike):
    @override
    def _reconstruct_isl_object(self) -> isl.Map:
        """
        Relies on the dimension type ordering in
        :func:`_deconstruct_isl_object`.
        """
        domain = isl.Set("{ [] }")
        range = self._obj

        map = isl.Map.from_domain_and_range(domain, range)

        param_start = self._parameter_dim_start
        map = map.move_dims(
            isl.dim_type.param, 0,
            isl.dim_type.set, param_start, len(self._parameter_names)
        )

        inp_start = self._input_dim_start - len(self._parameter_names)
        map = map.move_dims(
            isl.dim_type.in_, 0,
            isl.dim_type.set, inp_start, len(self._input_names)
        )

        return map


@overload
def make_map(src: str, ctx: isl.Context | None = None) -> Map:
    ...


@overload
def make_map(src: isl.Map) -> Map:
    ...


def make_map(src: str | isl.Map, ctx: isl.Context | None = None) -> Map:
    obj = isl.Map(src, ctx) if isinstance(src, str) else src

    set_obj, (param_names, inp_names) = _deconstruct_set_like_object(obj)
    set_obj, name_to_dim = _strip_names(set_obj)

    parameter_dim_start = min(
        name_to_dim[name]
        for name in name_to_dim
    )

    input_dim_start = min(
        name_to_dim[name]
        for name in name_to_dim
    )

    return Map(
        set_obj,
        name_to_dim,
        param_names,
        parameter_dim_start,
        _input_names=inp_names,
        _input_dim_start=input_dim_start
    )
