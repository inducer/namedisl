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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Generic, TypeAlias, TypeVar

from constantdict import constantdict
from typing_extensions import override

import islpy as isl


IslSetLike = isl.BasicSet | isl.BasicMap | isl.Set | isl.Map
IslBaseExpressionLike = isl.Aff | isl.QPolynomial
IslPwExpressionLike = isl.PwAff | isl.PwQPolynomial
IslMultiExpressionLike = isl.MultiAff | isl.PwMultiAff
IslExpressionLike = IslBaseExpressionLike | IslPwExpressionLike | IslMultiExpressionLike

IslExpressionLikeT = TypeVar(
    "IslExpressionLikeT",
    isl.Aff,
    isl.MultiAff,
    isl.PwAff,
    isl.PwMultiAff,
    isl.QPolynomial,
    isl.PwQPolynomial
)
IslSetLikeT = TypeVar(
    "IslSetLikeT",
    isl.BasicSet,
    isl.BasicMap,
    isl.Set,
    isl.Map
)
IslObjectT = TypeVar("IslObjectT", IslSetLike, IslExpressionLike)

NameToDim: TypeAlias = Mapping[str, int]

# NOTE: without tracking what dimension type a particular name belongs to, it is
# not possible to reconstruct the ISL object after dimension operations, e.g.
# alignment
DimTypeToNames: TypeAlias = Mapping[isl.dim_type, frozenset[str]]

IslObjectPieces: TypeAlias = tuple[IslSetLike | IslExpressionLike, DimTypeToNames]


__version__ = metadata.version("namedisl")
_match = re.match(r"^([0-9.]+)([a-z0-9]*?)$", __version__)
assert _match
VERSION = tuple(int(nr) for nr in _match.group(1).split("."))

__all__ = [
    "_align_and_apply_binary_op",
    "_align_two",
    "_deconstruct_object",
    "_find_contiguous_dim_chunks",
    "_restore_names",
    "_strip_names",
]


def _strip_names(obj: IslObjectT) -> tuple[IslObjectT, NameToDim]:
    name_to_dim: dict[str, int] = {}
    for i in range(obj.dim(isl.dim_type.set)):

        if isinstance(obj, isl.QPolynomial | isl.PwQPolynomial):
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


def _deconstruct_object(obj: IslObjectT) -> IslObjectPieces:
    from islpy import dim_type

    dt_to_names: dict[dim_type, frozenset[str]] = {}

    if isinstance(obj, IslSetLike):
        setlike_obj = obj
        dt_to_names = dict.fromkeys(
            [isl.dim_type.in_, isl.dim_type.param], frozenset()
        )
        for dt in dt_to_names:
            dt_to_names[dt] = _get_dim_names(setlike_obj, dt)
            if dt_to_names[dt]:
                setlike_obj = setlike_obj.move_dims(
                    dim_type.set,
                    setlike_obj.dim(dim_type.set),
                    dt,
                    0,
                    setlike_obj.dim(dt)
                )

        setlike_obj = (
            setlike_obj.range()
            if isinstance(setlike_obj, isl.Map)
            else setlike_obj
        )

        return setlike_obj, constantdict(dt_to_names)

    elif isinstance(obj, IslExpressionLike):
        expr_obj = obj

        dt_to_names = dict.fromkeys([isl.dim_type.param], frozenset())

        return expr_obj, constantdict(dt_to_names)


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
    all_names = [
        *sorted(all_set_names),
        *sorted(all_inp_names),
        *sorted(all_param_names),
    ]

    name_to_dim: NameToDim = constantdict({
        name: pos for pos, name in enumerate(all_names)
    })

    return name_to_dim, constantdict(dt_to_names)


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

    @abstractmethod
    def _reconstruct_isl_object(self) -> IslExpressionLike | IslSetLike:
        ...

    @override
    def __str__(self) -> str:
        return str(self._reconstruct_isl_object())
