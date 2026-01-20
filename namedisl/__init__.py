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
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
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


IslSetLike = isl.Set | isl.Map
IslExpressionLike = isl.Aff | isl.QPolynomial

IslExpressionLikeT = TypeVar("IslExpressionLikeT", isl.Aff, isl.QPolynomial)
IslSetLikeT = TypeVar("IslSetLikeT", isl.Set, isl.Map)
IslObjectT = TypeVar("IslObjectT", IslSetLike, IslExpressionLike)

NameToDim: TypeAlias = Mapping[str, int]

# NOTE: without tracking what dimension type a particular name belongs to, it is
# not possible to reconstruct the ISL object after dimension operations, e.g.
# alignment
DimTypeToNames: TypeAlias = Mapping[isl.dim_type, frozenset[str]]

SetLikePieces: TypeAlias = tuple[isl.Set, DimTypeToNames]


def _strip_names(obj: IslObjectT) -> tuple[IslObjectT, NameToDim]:
    name_to_dim: dict[str, int] = {}
    for i in range(obj.dim(isl.dim_type.set)):

        if isinstance(obj, isl.QPolynomial):
            name = obj.space.get_dim_name(isl.dim_type.set, i)
        else:
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

        if isinstance(obj, isl.QPolynomial):
            name = obj.space.get_dim_name(dt, dim)
        else:
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


def _find_contiguous_dim_chunks(dims: Sequence[int]) -> Mapping[int, int]:
    """
    Determines contiguous chunks of dimensions within a sequence of dimensions.
    Returns a mapping of the first dimension in the chunk to the length of the
    chunk.
    """
    if not dims:
        return {}

    chunks: dict[int, int] = {}

    start = dims[0]
    count = 1

    from itertools import pairwise
    for prev, curr in pairwise(dims):
        if curr == prev + 1:
            count += 1
        else:
            chunks[start] = count
            start = curr
            count = 1

    chunks[start] = count

    return constantdict(chunks)


# FIXME: think through whether or not alphabetical ordering will require more
# work on average than using one of the objects as a template in alignment
def _find_joint_name_to_dim(
        obj1: NamedIslObject[IslObjectT],
        obj2: NamedIslObject[IslObjectT]
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
    all_names  = sorted(all_set_names)
    all_names += sorted(all_inp_names)
    all_names += sorted(all_param_names)

    name_to_dim: NameToDim = {}
    for pos, name in enumerate(all_names):
        name_to_dim[name] = pos

    return constantdict(name_to_dim), constantdict(dt_to_names)


def _align_obj(
        named_obj: NamedIslObject[IslObjectT],
        ordering: NameToDim,
        dimtype_to_names: DimTypeToNames
    ) -> NamedIslObject[IslObjectT]:
    new_isl_obj = named_obj._obj
    running_name_to_dim = dict(named_obj._name_to_dim)

    for name, target_dim in sorted(ordering.items(), key=lambda x: x[1]):
        if name in running_name_to_dim:
            old_dim = running_name_to_dim[name]

            if old_dim == target_dim:
                continue

            # temporarily move to parameter dimension since destination and
            # source dim types cannot match in ISL
            new_isl_obj = new_isl_obj.move_dims(
                isl.dim_type.param, 0,
                isl.dim_type.set, old_dim, 1
            )

            new_isl_obj = new_isl_obj.move_dims(
                isl.dim_type.set, target_dim,
                isl.dim_type.param, 0, 1
            )

        else:
            old_dim = new_isl_obj.dim(isl.dim_type.set)
            new_isl_obj = new_isl_obj.insert_dims(isl.dim_type.set, target_dim, 1)

        # track side effects of inserting/swapping dimensions
        for n, d in list(running_name_to_dim.items()):
            if (target_dim > old_dim) and (d > old_dim):
                running_name_to_dim[n] = d - 1
            elif (target_dim < old_dim) and (d < old_dim):
                running_name_to_dim[n] = d + 1

        running_name_to_dim[name] = target_dim

    return type(named_obj)(new_isl_obj, ordering, dimtype_to_names)


def _align_two(
        named_obj1: NamedIslObject[IslObjectT],
        named_obj2: NamedIslObject[IslObjectT]
    ) -> tuple[NamedIslObject[IslObjectT], ...]:

    name_to_dim, dimtype_to_names = _find_joint_name_to_dim(named_obj1,
                                                            named_obj2)

    named_obj1 = _align_obj(named_obj1, name_to_dim, dimtype_to_names)
    named_obj2 = _align_obj(named_obj2, name_to_dim, dimtype_to_names)

    return named_obj1, named_obj2


def _align_and_apply_binary_op(
        lhs: NamedIslObject[IslObjectT],
        rhs: NamedIslObject[IslObjectT],
        op:  Callable[[IslObjectT, IslObjectT], IslObjectT]
    ) -> NamedIslObject[IslObjectT]:

    lhs, rhs = _align_two(lhs, rhs)
    result = op(lhs._obj, rhs._obj)

    # NOTE: since lhs and rhs were aligned, they both agree on what name-to-dim
    # and dimtype-to-name is, can just take information from lhs
    return type(lhs)(result, lhs._name_to_dim, lhs._dimtype_to_names)


@dataclass(frozen=True, eq=False)
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

    def __and__(
        self, other: NamedIslObject[IslObjectT]) -> NamedIslObject[IslObjectT]:
        return _align_and_apply_binary_op(self, other, operator.and_)

    def __or__(
            self, other: NamedIslObject[IslObjectT]) -> NamedIslObject[IslObjectT]:
        return _align_and_apply_binary_op(self, other, operator.or_)

    def __sub__(
            self, other: NamedIslObject[IslObjectT]) -> NamedIslObject[IslObjectT]:
        return _align_and_apply_binary_op(self, other, operator.sub)

    @abstractmethod
    def _reconstruct_isl_object(self) -> IslExpressionLike | IslSetLike:
        ...

    @override
    def __str__(self) -> str:
        return str(self._reconstruct_isl_object())


@dataclass(frozen=True, eq=False)
class _NamedIslSetLike(NamedIslObject[isl.Set], ABC):
    """
    Represents set-like objects with parameter dimensions as a non-parameterized
    set. Names are organized as contiguous chunks of dimension types, i.e.
        [ (set names), (input names), (parameter names) ]
    """
    _obj: isl.Set

    def complement(self: _NamedIslSetLike) -> _NamedIslSetLike:
        return type(self)(self._obj.complement(),
                          self._name_to_dim,
                          self._dimtype_to_names)

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

        return type(self)(
            new_isl_obj,
            self._name_to_dim,  # NOTE: no dimensions are removed by elimination
            self._dimtype_to_names
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

        return type(self)(
            new_isl_obj,
            constantdict(new_name_to_dim),
            new_type_to_names
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

    # {{{ TODO: funtions that return ExpressionLike objects

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

    set_obj, dimtype_to_names = _deconstruct_set_like_object(obj)
    set_obj, name_to_dim = _strip_names(set_obj)

    assert isinstance(set_obj, isl.Set)
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

    set_obj, dimtype_to_names = _deconstruct_set_like_object(obj)
    set_obj, name_to_dim = _strip_names(set_obj)

    assert isinstance(set_obj, isl.Set)
    return Map(set_obj, name_to_dim, dimtype_to_names)
