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
from dataclasses import dataclass
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

NameToDim: TypeAlias = Mapping[str, int]

# NOTE: without tracking what dimension type a particular name belongs to, it is
# not possible to reconstruct the ISL object after dimension operations, e.g.
# alignment
DimTypeToNames: TypeAlias = Mapping[isl.dim_type, frozenset[str]]

SetLikePieces: TypeAlias = tuple[isl.Set, DimTypeToNames]


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


def _restore_names(obj: IslObjectT, name_to_dim: NameToDim) -> IslObjectT:
    for name, dim in name_to_dim.items():
        obj = obj.set_dim_name(isl.dim_type.set, dim, name)
    return obj


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

    dt_to_names: dict[dim_type, frozenset[str]] = dict.fromkeys(
        [isl.dim_type.in_, isl.dim_type.param], frozenset()
    )
    for dt in dt_to_names:
        dt_to_names[dt] = _get_dim_names(obj, dt)
        if dt_to_names[dt]:
            obj = obj.move_dims(
                dim_type.set,
                obj.dim(dim_type.set),
                dt,
                0,
                obj.dim(dt)
            )

    set_obj = obj.range() if isinstance(obj, isl.Map) else obj

    return set_obj, constantdict(dt_to_names)


@dataclass(frozen=True)
class NamedIslObject(Generic[IslObjectT], ABC):
    _obj: IslObjectT
    _name_to_dim: NameToDim

    # used to reconstruct ISL object
    _dimtype_to_names: DimTypeToNames

    @property
    def _has_inputs(self) -> bool:
        return (
            isl.dim_type.in_ in self._dimtype_to_names
            and
            len(self._dimtype_to_names[isl.dim_type.in_]) > 0
        )

    @property
    def _input_names(self) -> frozenset[str]:
        if self._has_inputs:
            return self._dimtype_to_names[isl.dim_type.in_]
        return frozenset()

    @property
    def _input_dim_start(self) -> int | None:
        if self._has_inputs:
            return min(
                self._name_to_dim[name]
                for name in self._dimtype_to_names[isl.dim_type.in_]
            )
        return None

    @property
    def _has_params(self) -> bool:
        return (
            isl.dim_type.param in self._dimtype_to_names
            and
            len(self._dimtype_to_names[isl.dim_type.param]) > 0
        )

    @property
    def _parameter_names(self) -> frozenset[str]:
        if self._has_params:
            return self._dimtype_to_names[isl.dim_type.param]
        return frozenset()

    @property
    def _parameter_dim_start(self) -> int | None:
        if self._has_params:
            return min(
                self._name_to_dim[name]
                for name in self._dimtype_to_names[isl.dim_type.param]
            )
        return None

    @abstractmethod
    def _reconstruct_isl_object(self) -> IslExpressionLike | IslSetLike:
        ...

    @override
    def __str__(self) -> str:
        return str(self._reconstruct_isl_object())


@dataclass(frozen=True)
class _NamedIslSetLike(NamedIslObject[isl.Set], ABC):
    """
    Represents set-like objects with parameter dimensions as a non-parameterized
    set. Names are organized as contiguous chunks of dimension types, i.e.
        [ (set names), (input names), (parameter names) ]
    """
    _obj: isl.Set


# FIXME: think through whether or not alphabetical ordering will require more
# work on average than using one of the objects as a template in alignment
def _find_joint_name_to_dim(
        obj1: _NamedIslSetLike,
        obj2: _NamedIslSetLike
    ) -> tuple[NameToDim, DimTypeToNames]:
    """
    Enforces alphabetical ordering of all dimensions found in :arg:`obj1` and
    :arg:`obj2` within each dimension-type chunk. This ordering is used in
    alignment before performing operations between two set-like objects.
    """
    obj1_inp_names = obj1._input_names
    obj1_param_names = obj1._parameter_names
    obj1_set_names = (
        frozenset(obj1._name_to_dim.keys()) - (obj1_inp_names | obj1_param_names)
    )

    obj2_inp_names = obj2._input_names
    obj2_param_names = obj2._parameter_names
    obj2_set_names = (
        frozenset(obj2._name_to_dim.keys()) - (obj2_inp_names | obj2_param_names)
    )

    all_inp_names = obj1_inp_names | obj2_inp_names
    all_param_names = obj1_param_names | obj2_param_names
    all_set_names = obj1_set_names | obj2_set_names

    dt_to_names: DimTypeToNames = {}
    dt_to_names[isl.dim_type.param] = all_param_names
    dt_to_names[isl.dim_type.in_] = all_inp_names

    # enforces contiguous ordering of [ (set), (input), (param) ] in set
    # representation
    all_names  = sorted(list(all_set_names))
    all_names += sorted(list(all_inp_names))
    all_names += sorted(list(all_param_names))

    name_to_dim: NameToDim = {}
    for pos, name in enumerate(all_names):
        name_to_dim[name] = pos

    return constantdict(name_to_dim), constantdict(dt_to_names)


def _align_obj(
        named_obj: _NamedIslSetLike,
        ordering: NameToDim,
        dimtype_to_names: DimTypeToNames
    ) -> _NamedIslSetLike:
    new_isl_obj = named_obj._obj
    running_name_to_dim = dict(named_obj._name_to_dim)

    # three cases for alignment
    # - if name does not currently exist, then add a dimension
    # - if name exists:
    #   - if new dim and old dim match, then do nothing
    #   - if new dim and old dim do not match, then swap and update old dim

    for name, dim in sorted(ordering.items(), key=lambda x: x[1]):
        if name in running_name_to_dim:
            old_dim = running_name_to_dim[name]

            if old_dim == dim:
                continue

            # temporarily move to parameter dimension since destination and
            # source dim types cannot match in ISL
            new_isl_obj = new_isl_obj.move_dims(
                isl.dim_type.param, 0,
                isl.dim_type.set, dim, 1
            )

            new_isl_obj = new_isl_obj.move_dims(
                isl.dim_type.set, dim,
                isl.dim_type.param, 0, 1
            )

        else:
            old_dim = new_isl_obj.dim(isl.dim_type.set)
            new_isl_obj = new_isl_obj.insert_dims(isl.dim_type.set, dim, 1)

        # track side effects of inserting/swapping dimensions
        temp_name_to_dim = running_name_to_dim.copy()
        for cur_name, cur_dim in sorted(
                running_name_to_dim.items(), key=lambda x: x[1]):
            if (dim > old_dim) and (cur_dim > old_dim):
                temp_name_to_dim[cur_name] = cur_dim - 1
            elif (dim < old_dim) and (cur_dim < old_dim):
                temp_name_to_dim[cur_name] = cur_dim + 1

        running_name_to_dim = temp_name_to_dim
        running_name_to_dim[name] = dim

    return type(named_obj)(new_isl_obj, ordering, dimtype_to_names)


def _align_two(named_obj1: _NamedIslSetLike,
               named_obj2: _NamedIslSetLike) -> tuple[_NamedIslSetLike, ...]:

    name_to_dim, dimtype_to_names = _find_joint_name_to_dim(named_obj1,
                                                            named_obj2)

    named_obj1 = _align_obj(named_obj1, name_to_dim, dimtype_to_names)
    named_obj2 = _align_obj(named_obj2, name_to_dim, dimtype_to_names)

    return named_obj1, named_obj2



@final
@dataclass(frozen=True, eq=False)
class Set(_NamedIslSetLike):
    @override
    def _reconstruct_isl_object(self) -> isl.Set:
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


@overload
def make_set(src: str, ctx: isl.Context | None = None) -> Set:
    ...


@overload
def make_set(src: isl.Set) -> Set:
    ...


def make_set(src: isl.Set | str, ctx: isl.Context | None = None) -> Set:
    obj = isl.Set(src, ctx) if isinstance(src, str) else src

    set_obj, dimtype_to_names = _deconstruct_set_like_object(obj)
    set_obj, name_to_dim = _strip_names(set_obj)

    return Set(set_obj, name_to_dim, dimtype_to_names)


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


@overload
def make_map(src: str, ctx: isl.Context | None = None) -> Map:
    ...


@overload
def make_map(src: isl.Map) -> Map:
    ...


def make_map(src: str | isl.Map, ctx: isl.Context | None = None) -> Map:
    obj = isl.Map(src, ctx) if isinstance(src, str) else src

    set_obj, dimtype_to_names = _deconstruct_set_like_object(obj)
    set_obj, name_to_dim = _strip_names(set_obj)

    return Map(set_obj, name_to_dim, dimtype_to_names)
