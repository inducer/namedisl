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
from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar, overload

from constantdict import constantdict
from typing_extensions import Self, override

import islpy as isl


ISL_DIM_TYPES = [
    isl.dim_type.out,
    isl.dim_type.in_,
    isl.dim_type.set,
    isl.dim_type.param
]


if TYPE_CHECKING:
    from namedisl.tags import _TaggedName


IslBaseExpressionLike = isl.Aff | isl.QPolynomial
IslPwExpressionLike = isl.PwAff | isl.PwQPolynomial
IslMultiExpressionLike = isl.MultiAff | isl.PwMultiAff

IslExpressionLike = IslBaseExpressionLike | IslPwExpressionLike | IslMultiExpressionLike
IslSetLike = isl.BasicSet | isl.BasicMap | isl.Set | isl.Map
IslObject = IslSetLike | IslExpressionLike | IslMultiExpressionLike

IslExpressionLikeT = TypeVar(
    "IslExpressionLikeT",
    bound=IslExpressionLike,
)
IslSetLikeT = TypeVar(
    "IslSetLikeT",
    bound=IslSetLike
)
IslMultiExpressionLikeT = TypeVar(
    "IslMultiExpressionLikeT",
    bound=IslMultiExpressionLike
)
IslPwExpressionLikeT = TypeVar(
    "IslPwExpressionLikeT",
    bound=IslPwExpressionLike
)
IslObjectT = TypeVar("IslObjectT", bound=IslObject)

NamedIslObjectT = TypeVar("NamedIslObjectT", bound="NamedIslObject[IslObject]")

NameToDim: TypeAlias = Mapping[str, int]

# NOTE: without tracking what dimension type a particular name belongs to, it is
# not possible to reconstruct the ISL object after dimension operations, e.g.
# alignment
DimTypeToNames: TypeAlias = Mapping[isl.dim_type, frozenset[str]]

IslObjectPieces: TypeAlias = tuple[IslObjectT, DimTypeToNames]


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

    dt_to_strip = (
        isl.dim_type.set if isinstance(obj, IslSetLike) else isl.dim_type.in_
    )

    for i in range(obj.dim(dt_to_strip)):
        if isinstance(obj, isl.QPolynomial | isl.PwQPolynomial):
            name = obj.space.get_dim_name(dt_to_strip, i)
        else:
            name = obj.get_dim_name(dt_to_strip, i)

        if name is None:
            raise ValueError("unnamed dimension found")

        if name in name_to_dim:
            raise ValueError(f"non-unique dim name: {name}")

        name_to_dim[name] = i

    return obj, constantdict(name_to_dim)


@overload
def _restore_names(obj: isl.PwAff, name_to_dim: NameToDim) -> isl.PwAff:
    ...


@overload
def _restore_names(obj: IslSetLikeT, name_to_dim: NameToDim) -> IslSetLikeT:
    ...


@overload
def _restore_names(
            obj: IslPwExpressionLikeT,
            name_to_dim: NameToDim
        ) -> IslPwExpressionLikeT:
    ...


def _restore_names(obj: IslObjectT, name_to_dim: NameToDim) -> IslObjectT:
    restored_obj = obj.copy()
    if isinstance(restored_obj, isl.PwAff):
        # input dimensions cannot be renamed for isl.PwAff, so we first move
        # input dims to the parameter space, rename then move back
        restored_obj = restored_obj.move_dims(
            isl.dim_type.param,
            0,
            isl.dim_type.in_,
            0,
            restored_obj.dim(isl.dim_type.in_)
        )

        for name, dim in name_to_dim.items():
            restored_obj = restored_obj.set_dim_name(
                isl.dim_type.param,
                dim,
                name
            )

        restored_obj = restored_obj.get_pw_aff_list().get_at(0)
        return restored_obj.move_dims(
            isl.dim_type.in_,
            0,
            isl.dim_type.param,
            0,
            restored_obj.dim(isl.dim_type.param)
        )

    if isinstance(restored_obj, IslSetLike):
        dt_to_restore = isl.dim_type.set
    else:
        dt_to_restore = isl.dim_type.in_

    for name, dim in name_to_dim.items():
        restored_obj = restored_obj.set_dim_name(dt_to_restore, dim, name)

    if isinstance(restored_obj, isl.UnionPwAff | isl.UnionPwMultiAff):
        raise NotImplementedError

    return restored_obj


def _get_dim_names(obj: IslObject, dt: isl.dim_type) -> frozenset[str]:
    all_dt_names: list[str] = []
    for dim in range(obj.dim(dt)):

        if isinstance(obj, isl.QPolynomial | isl.PwQPolynomial):
            name = obj.space.get_dim_name(dt, dim)
        else:
            name = obj.get_dim_name(dt, dim)

        if name is None:
            raise ValueError("unnamed dimension found")

        all_dt_names.append(name)

    return frozenset(all_dt_names)


@overload
def _deconstruct_object(obj: isl.Map) -> tuple[isl.Set, DimTypeToNames]:
    ...


@overload
# PwMultiAff doesn't have move_dims, so we're being a bit crooked here.
def _deconstruct_object(obj: isl.PwMultiAff) -> tuple[isl.Set, DimTypeToNames]:
    ...


@overload
def _deconstruct_object(obj: IslObjectT) -> tuple[IslObjectT, DimTypeToNames]:
    ...


def _deconstruct_object(obj: IslObjectT) -> tuple[IslObject, DimTypeToNames]:
    dt_to_names: dict[isl.dim_type, frozenset[str]] = {}

    if isinstance(obj, IslSetLike | IslMultiExpressionLike):
        decon_obj = obj
        dt_to_names = dict.fromkeys(
            [isl.dim_type.in_, isl.dim_type.param], frozenset()
        )

        # NOTE: isl.PwMultiAff.move_dims does not exist, represent as map
        # internally
        if isinstance(decon_obj, IslMultiExpressionLike):
            decon_obj = decon_obj.as_map()

        for dt in dt_to_names:
            dt_to_names[dt] = _get_dim_names(decon_obj, dt)
            if dt_to_names[dt]:
                decon_obj = decon_obj.move_dims(
                    isl.dim_type.set,
                    decon_obj.dim(isl.dim_type.set),
                    dt,
                    0,
                    decon_obj.dim(dt)
                )

        decon_obj = (
            decon_obj.range()
            if isinstance(decon_obj, isl.Map | isl.BasicMap)
            else decon_obj
        )

        decon_obj = (
            isl.Set.from_basic_set(decon_obj)
            if isinstance(decon_obj, isl.BasicSet)
            else decon_obj
        )

    else:
        decon_obj = obj

        dt_to_names = dict.fromkeys([isl.dim_type.param], frozenset())
        dt_to_names[isl.dim_type.param] = _get_dim_names(decon_obj,
                                                         isl.dim_type.param)

        decon_obj = decon_obj.move_dims(
            isl.dim_type.in_,
            decon_obj.dim(isl.dim_type.in_),
            isl.dim_type.param,
            0,
            decon_obj.dim(isl.dim_type.param)
        )

    return decon_obj, constantdict(dt_to_names)


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
    obj1_inp_names = obj1.input_names
    obj1_param_names = obj1.parameter_names
    obj1_set_names = (
        frozenset(obj1._name_to_dim.keys()) - (obj1_inp_names | obj1_param_names)
    )

    obj2_inp_names = obj2.input_names
    obj2_param_names = obj2.parameter_names
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
        named_obj: NamedIslObjectT,
        ordering: NameToDim,
        dimtype_to_names: DimTypeToNames
    ) -> NamedIslObjectT:
    new_isl_obj = named_obj._obj
    running_name_to_dim = dict(named_obj._name_to_dim)

    target_dt = (
        isl.dim_type.set
        if isinstance(new_isl_obj, IslSetLike)
        else isl.dim_type.in_
    )

    for name, target_dim in sorted(ordering.items(), key=lambda x: x[1]):
        if name in running_name_to_dim:
            old_dim = running_name_to_dim[name]

            if old_dim == target_dim:
                continue

            # temporarily move to parameter dimension since destination and
            # source dim types cannot match in ISL
            new_isl_obj = new_isl_obj.move_dims(
                isl.dim_type.param, 0,
                target_dt, old_dim, 1
            )

            new_isl_obj = new_isl_obj.move_dims(
                target_dt, target_dim,
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
        named_obj1: NamedIslObjectT,
        named_obj2: NamedIslObjectT
    ) -> tuple[NamedIslObjectT, ...]:

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
class NamedIslObject(ABC, Generic[IslObjectT]):
    _obj: IslObjectT
    _name_to_dim: NameToDim

    # used to reconstruct ISL object
    _dimtype_to_names: DimTypeToNames

    def add_names(self, tagged_names_to_add: Sequence[_TaggedName]) -> Self:

        if isinstance(self._obj, isl.PwMultiAff):
            raise NotImplementedError

        new_obj = self._obj
        new_name_to_dim = dict(self._name_to_dim)
        new_dt_to_names: Mapping[isl.dim_type, frozenset[str]] = dict.fromkeys(
            ISL_DIM_TYPES, frozenset()
        )

        for tagged_name in tagged_names_to_add:
            name = tagged_name.name
            dt = tagged_name._isl_dim_type

            new_dt_to_names[dt] |= frozenset({name})

        # get rid of unused keys
        new_dt_to_names = {
            dt: new_dt_to_names[dt]
            for dt in new_dt_to_names if new_dt_to_names[dt]
        }

        for dt in new_dt_to_names:
            if dt in (isl.dim_type.out, isl.dim_type.set):
                start = 0
            elif dt == isl.dim_type.in_:
                start = self._input_dim_start
            else:
                start = self._parameter_dim_start

            new_obj = new_obj.insert_dims(dt, start, len(new_dt_to_names[dt]))

        return type(self)(
            new_obj,
            constantdict(new_name_to_dim),
            constantdict(new_dt_to_names))

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._name_to_dim.keys())

    @property
    def _has_inputs(self) -> bool:
        return (
            isl.dim_type.in_ in self._dimtype_to_names
            and
            len(self._dimtype_to_names[isl.dim_type.in_]) > 0
        )

    @property
    def input_names(self) -> frozenset[str]:
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
    def parameter_names(self) -> frozenset[str]:
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

    def _reconstruct_isl_object(self) -> IslObjectT:
        """
        Relies on the dimension type ordering in
        :func:`_deconstruct_set_like_object`.
        """
        obj = _restore_names(self._obj, self._name_to_dim)

        internal_dim = (
            isl.dim_type.set if isinstance(obj, isl.Set) else isl.dim_type.in_
        )

        if self._has_params:
            if self._parameter_dim_start is None:
                raise ValueError(
                    "Object has parameter dimensions, but a starting index for "
                    "parameter names is not given. Reconstruction is not "
                    "possible")

            param_start = self._parameter_dim_start
            obj = obj.move_dims(
                isl.dim_type.param, 0,
                internal_dim, param_start, len(self.parameter_names)
            )

        if self._has_inputs:
            if self._input_dim_start is None:
                raise ValueError(
                    "Object has input dimensions, but a starting index for "
                    "input names is not given. Reconstruction is not "
                    "possible")

            obj_domain = isl.Set("{ [] }")
            obj_range = obj

            obj = isl.Map.from_domain_and_range(obj_domain, obj_range)

            inp_start = self._input_dim_start
            obj = obj.move_dims(
                isl.dim_type.in_, 0,
                internal_dim, inp_start, len(self.input_names)
            )

        return obj

    @override
    def __str__(self) -> str:
        return str(self._reconstruct_isl_object())
