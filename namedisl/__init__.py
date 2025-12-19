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
from abc import ABC, abstractmethod
from collections.abc import Mapping
from importlib import metadata
from typing import Generic, TypeAlias, TypeVar, final

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

IslObjectPieces: TypeAlias = tuple[IslObjectT, tuple[frozenset[str], ...]]

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


def _move_dims_to_set_dim(obj: IslObjectT, dt: isl.dim_type) -> IslObjectT:
    obj = obj.move_dims(
        isl.dim_type.set, obj.dim(isl.dim_type.set),
        dt, 0, obj.dim(dt)
    )

    return obj


class NamedIslObject(Generic[IslObjectT], ABC):
    _obj: IslObjectT
    _name_to_dim: NameToDim
    _parameter_names: frozenset[str]
    _parameter_dim_start: int

    @abstractmethod
    def __init__(self, src: IslObjectT | str, ctx: isl.Context | None = None):
        ...

    # FIXME: needs a type on obj relaxed enough to be specialized in subclasses
    @abstractmethod
    def _deconstruct_isl_object(self, obj) -> IslObjectPieces[IslObjectT]:
        ...

    @abstractmethod
    def _reconstruct_isl_object(self) -> IslExpressionLike | IslSetLike:
        ...

    @override
    def __str__(self) -> str:
        return str(self._reconstruct_isl_object())


class _NamedIslSetLike(NamedIslObject[isl.Set], ABC):
    _obj: isl.Set


@final
class Set(_NamedIslSetLike):
    def __init__(self, src: isl.Set | str, ctx: isl.Context | None = None):
        obj = isl.Set(src, ctx) if isinstance(src, str) else src

        obj, (parameter_names,) = self._deconstruct_isl_object(obj)
        obj, name_to_dim = _strip_names(obj)

        self._obj = obj
        self._name_to_dim = name_to_dim
        self._parameter_names = parameter_names

        self._parameter_dim_start = min(
            self._name_to_dim[name]
            for name in self._parameter_names
        )

    @override
    def _deconstruct_isl_object(self, obj: isl.Set) -> IslObjectPieces[isl.Set]:
        """
        Internal set dimensions ordered in two contiguous chunks:
        [ (set dimensions), (parameter dimensions) ]
        """
        parameter_names = _get_dim_names(obj, isl.dim_type.param)
        obj = _move_dims_to_set_dim(obj, isl.dim_type.param)
        return obj, (frozenset(parameter_names),)

    @override
    def _reconstruct_isl_object(self) -> isl.Set:
        return self._obj.move_dims(
            isl.dim_type.param, 0,
            isl.dim_type.set, self._parameter_dim_start,
                len(self._parameter_names)
        )


@final
class Map(_NamedIslSetLike):
    _input_names: frozenset[str]
    _input_dim_start: int

    def __init__(self, src: isl.Map | str, ctx: isl.Context | None = None):
        obj = isl.Map(src, ctx) if isinstance(src, str) else src

        obj, (parameter_names, input_names) = self._deconstruct_isl_object(obj)
        obj, name_to_dim = _strip_names(obj)

        self._parameter_names = parameter_names
        self._input_names = input_names
        self._name_to_dim = name_to_dim
        self._obj = obj

        self._parameter_dim_start = min(
            self._name_to_dim[name]
            for name in self._parameter_names
        )

        self._input_dim_start = min(
            self._name_to_dim[name]
            for name in self._input_names
        )

        # NOTE: hard requirement for object reconstruction is to have each type
        # of dimension contiguous in the underlying set. each type of dimension
        # can be shuffled around arbitrarily within each contiguous chunk.
        # impose chunk ordering as [ (set), (parameter), (input) ]
        if self._input_dim_start < self._parameter_dim_start:
            raise ValueError(
                "Expected input dimensions to be ordered after parameter "
                "dimensions in set representation"
            )

    @override
    def _deconstruct_isl_object(self, obj: isl.Map) -> IslObjectPieces[isl.Set]:
        """
        Internal set dimensions ordered in three contiguous chunks as:
        [ (set dimensions), (parameter dimensions), (input dimensions) ]
        """
        parameter_names = _get_dim_names(obj, isl.dim_type.param)
        input_names = _get_dim_names(obj, isl.dim_type.in_)

        obj = _move_dims_to_set_dim(obj, isl.dim_type.param)
        obj = _move_dims_to_set_dim(obj, isl.dim_type.in_)

        return obj.range(), (parameter_names, input_names)

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
