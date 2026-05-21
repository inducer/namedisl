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
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar, cast, overload

from constantdict import constantdict
from typing_extensions import Self, override

import islpy as isl


ISL_DIM_TYPES = [
    isl.dim_type.out,
    isl.dim_type.in_,
    isl.dim_type.set,
    isl.dim_type.param,
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
IslSetLikeT = TypeVar("IslSetLikeT", bound=IslSetLike)
IslMultiExpressionLikeT = TypeVar(
    "IslMultiExpressionLikeT", bound=IslMultiExpressionLike
)
IslPwExpressionLikeT = TypeVar("IslPwExpressionLikeT", bound=IslPwExpressionLike)
IslObjectT = TypeVar("IslObjectT", bound=IslObject, covariant=True)  # noqa: PLC0105

NamedIslObjectT = TypeVar("NamedIslObjectT", bound="NamedIslObject[IslObject]")

NameToDim: TypeAlias = Mapping[str, int]

# NOTE: without tracking what dimension type a particular name belongs to, it is
# not possible to reconstruct the ISL object after dimension operations, e.g.
# alignment
DimTypeToNames: TypeAlias = Mapping[isl.dim_type, frozenset[str]]

IslObjectPieces: TypeAlias = tuple[IslObject, NameToDim, DimTypeToNames]


__version__ = metadata.version("namedisl")
_match = re.match(r"^([0-9.]+)([a-z0-9]*?)$", __version__)
assert _match
VERSION = tuple(int(nr) for nr in _match.group(1).split("."))

__all__ = [
    "_align_and_apply_binary_op",
    "_align_two",
    "_deconstruct_object",
    "_find_contiguous_dim_chunks",
    "_make_named_object_pieces",
    "_normalize_dimtype_to_names",
    "_restore_names",
    "_strip_names",
]


def _normalize_public_dim_type(dim_type: isl.dim_type) -> isl.dim_type:
    if dim_type == isl.dim_type.out:
        return isl.dim_type.set
    return dim_type


def _ensure_unique_public_names(obj: IslObject) -> None:
    if isinstance(obj, IslSetLike | IslMultiExpressionLike):
        dim_types = (isl.dim_type.set, isl.dim_type.in_, isl.dim_type.param)
    else:
        dim_types = (isl.dim_type.in_, isl.dim_type.param)

    seen_names: set[str] = set()
    for dim_type in dim_types:
        for dim in range(obj.dim(dim_type)):
            if isinstance(obj, isl.QPolynomial | isl.PwQPolynomial):
                name = obj.space.get_dim_name(dim_type, dim)
            else:
                name = obj.get_dim_name(dim_type, dim)
            if name is None:
                raise ValueError("duplicate or unnamed dimension found")
            if name in seen_names:
                raise ValueError(f"duplicate dimension name found: {name}")
            seen_names.add(name)


def _strip_names(obj: IslObjectT) -> tuple[IslObjectT, NameToDim]:
    name_to_dim: dict[str, int] = {}

    dt_to_strip = isl.dim_type.set if isinstance(obj, IslSetLike) else isl.dim_type.in_

    stripped_obj = obj.copy()

    for i in range(stripped_obj.dim(dt_to_strip)):
        if isinstance(stripped_obj, isl.QPolynomial | isl.PwQPolynomial):
            name = stripped_obj.space.get_dim_name(dt_to_strip, i)
        else:
            name = stripped_obj.get_dim_name(dt_to_strip, i)

        if name is None:
            raise ValueError("unnamed dimension found")

        if name in name_to_dim:
            raise ValueError(f"duplicate dimension name found: {name}")

        name_to_dim[name] = i

    return cast("IslObjectT", stripped_obj), constantdict(name_to_dim)


def _get_obj_dim_name(obj: IslObject, dt: isl.dim_type, dim: int) -> str:
    if isinstance(obj, isl.QPolynomial | isl.PwQPolynomial):
        name = obj.space.get_dim_name(dt, dim)
    else:
        name = obj.get_dim_name(dt, dim)

    if name is None:
        raise ValueError("unnamed dimension found")

    return name


def _normalize_dimtype_to_names(
    obj: IslObject, dimtype_to_names: DimTypeToNames
) -> DimTypeToNames:
    if isinstance(obj, IslSetLike | IslMultiExpressionLike):
        dim_type = isl.dim_type.set
        total_dims = obj.dim(dim_type)
        n_in = len(dimtype_to_names.get(isl.dim_type.in_, frozenset()))
        n_param = len(dimtype_to_names.get(isl.dim_type.param, frozenset()))

        new_dimtype_to_names: dict[isl.dim_type, frozenset[str]] = {}

        if n_in:
            start = total_dims - n_param - n_in
            new_dimtype_to_names[isl.dim_type.in_] = frozenset(
                _get_obj_dim_name(obj, dim_type, dim)
                for dim in range(start, start + n_in)
            )

        if n_param:
            start = total_dims - n_param
            new_dimtype_to_names[isl.dim_type.param] = frozenset(
                _get_obj_dim_name(obj, dim_type, dim)
                for dim in range(start, start + n_param)
            )

        return constantdict(new_dimtype_to_names)

    total_dims = obj.dim(isl.dim_type.in_)
    n_param = len(dimtype_to_names.get(isl.dim_type.param, frozenset()))
    if not n_param:
        return dimtype_to_names

    start = total_dims - n_param
    return constantdict({
        isl.dim_type.param: frozenset(
            _get_obj_dim_name(obj, isl.dim_type.in_, dim)
            for dim in range(start, start + n_param)
        )
    })


def _make_named_object_pieces(obj: IslObject) -> IslObjectPieces:
    _ensure_unique_public_names(obj)
    decon_obj, dimtype_to_names = _deconstruct_object(obj)
    decon_obj, name_to_dim = _strip_names(decon_obj)
    dimtype_to_names = _normalize_dimtype_to_names(decon_obj, dimtype_to_names)
    return decon_obj, name_to_dim, dimtype_to_names


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
            restored_obj.dim(isl.dim_type.in_),
        )

        for name, dim in name_to_dim.items():
            restored_obj = restored_obj.set_dim_name(isl.dim_type.param, dim, name)

        restored_obj = restored_obj.get_pw_aff_list().get_at(0)
        return cast(
            "IslObjectT",
            restored_obj.move_dims(
                isl.dim_type.in_,
                0,
                isl.dim_type.param,
                0,
                restored_obj.dim(isl.dim_type.param),
            ),
        )

    if isinstance(restored_obj, IslSetLike):
        dt_to_restore = isl.dim_type.set
    else:
        dt_to_restore = isl.dim_type.in_

    for name, dim in name_to_dim.items():
        restored_obj = restored_obj.set_dim_name(dt_to_restore, dim, name)

    if isinstance(restored_obj, isl.UnionPwAff | isl.UnionPwMultiAff):
        raise NotImplementedError

    return cast("IslObjectT", restored_obj)


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
def _deconstruct_object(obj: isl.Map) -> tuple[isl.Set, DimTypeToNames]: ...


@overload
# PwMultiAff doesn't have move_dims, so we're being a bit crooked here.
def _deconstruct_object(obj: isl.PwMultiAff) -> tuple[isl.Set, DimTypeToNames]: ...


@overload
def _deconstruct_object(obj: IslObject) -> tuple[IslObject, DimTypeToNames]: ...


def _deconstruct_object(obj: IslObject) -> tuple[IslObject, DimTypeToNames]:
    dt_to_names: dict[isl.dim_type, frozenset[str]] = {}

    if isinstance(obj, IslSetLike | IslMultiExpressionLike):
        decon_obj = obj
        dt_to_names = dict.fromkeys([isl.dim_type.in_, isl.dim_type.param], frozenset())

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
                    decon_obj.dim(dt),
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
        dt_to_names[isl.dim_type.param] = _get_dim_names(decon_obj, isl.dim_type.param)

        decon_obj = decon_obj.move_dims(
            isl.dim_type.in_,
            decon_obj.dim(isl.dim_type.in_),
            isl.dim_type.param,
            0,
            decon_obj.dim(isl.dim_type.param),
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
    obj1: NamedIslObject[IslObjectT], obj2: NamedIslObject[IslObjectT]
) -> tuple[NameToDim, DimTypeToNames]:
    """
    Enforces alphabetical ordering of all dimensions found in :arg:`obj1` and
    :arg:`obj2` within each dimension-type chunk. This ordering is used in
    alignment before performing operations between two set-like objects.
    """
    all_set_names = obj1._names_for_dim_type(
        isl.dim_type.set
    ) | obj2._names_for_dim_type(isl.dim_type.set)
    all_inp_names = obj1.input_names | obj2.input_names
    all_param_names = obj1.parameter_names | obj2.parameter_names

    dt_to_names: DimTypeToNames = {}
    dt_to_names[isl.dim_type.param] = all_param_names
    if isinstance(obj1._obj, IslSetLike | IslMultiExpressionLike) or isinstance(
        obj2._obj, IslSetLike | IslMultiExpressionLike
    ):
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
    named_obj: NamedIslObjectT, ordering: NameToDim, dimtype_to_names: DimTypeToNames
) -> NamedIslObjectT:
    new_isl_obj = cast(
        "IslSetLike | IslBaseExpressionLike | IslPwExpressionLike | isl.MultiAff",
        named_obj._obj,
    )
    running_name_to_dim = dict(named_obj._name_to_dim)

    target_dt = (
        isl.dim_type.set if isinstance(new_isl_obj, IslSetLike) else isl.dim_type.in_
    )

    for name, target_dim in sorted(ordering.items(), key=lambda x: x[1]):
        if name in running_name_to_dim:
            old_dim = running_name_to_dim[name]

            if old_dim == target_dim:
                continue

            # temporarily move to parameter dimension since destination and
            # source dim types cannot match in ISL
            new_isl_obj = new_isl_obj.move_dims(
                isl.dim_type.param, 0, target_dt, old_dim, 1
            )

            new_isl_obj = new_isl_obj.move_dims(
                target_dt, target_dim, isl.dim_type.param, 0, 1
            )

        else:
            old_dim = new_isl_obj.dim(target_dt)
            new_isl_obj = new_isl_obj.insert_dims(target_dt, target_dim, 1)

        # track side effects of inserting/swapping dimensions
        for n, d in list(running_name_to_dim.items()):
            if (target_dim > old_dim) and (d > old_dim):
                running_name_to_dim[n] = d - 1
            elif (target_dim < old_dim) and (d < old_dim):
                running_name_to_dim[n] = d + 1

        running_name_to_dim[name] = target_dim

    new_isl_obj = _restore_names(new_isl_obj, ordering)

    return type(named_obj)(
        new_isl_obj,
        ordering,
        dimtype_to_names,
    )


def _align_two(
    named_obj1: NamedIslObjectT, named_obj2: NamedIslObjectT
) -> tuple[NamedIslObjectT, ...]:

    name_to_dim, dimtype_to_names = _find_joint_name_to_dim(named_obj1, named_obj2)

    named_obj1 = _align_obj(named_obj1, name_to_dim, dimtype_to_names)
    named_obj2 = _align_obj(named_obj2, name_to_dim, dimtype_to_names)

    return named_obj1, named_obj2


def _align_and_apply_binary_op(
    lhs: NamedIslObject[IslObjectT],
    rhs: NamedIslObject[IslObjectT],
    op: Callable[[IslObjectT, IslObjectT], IslObjectT],
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

    @property
    def _metadata_input_names(self) -> frozenset[str]:
        return self._dimtype_to_names.get(isl.dim_type.in_, frozenset())

    @property
    def _metadata_parameter_names(self) -> frozenset[str]:
        return self._dimtype_to_names.get(isl.dim_type.param, frozenset())

    def _names_for_dim_type(self, dim_type: isl.dim_type) -> frozenset[str]:
        dim_type = _normalize_public_dim_type(dim_type)
        if dim_type == isl.dim_type.param:
            return self.parameter_names

        if isinstance(self._obj, IslSetLike | IslMultiExpressionLike):
            if dim_type == isl.dim_type.in_:
                return self._metadata_input_names
            if dim_type == isl.dim_type.set:
                return self.names - self._metadata_input_names - self.parameter_names
        else:
            if dim_type == isl.dim_type.in_:
                return self.names - self.parameter_names
            if dim_type == isl.dim_type.set:
                return frozenset()

        raise ValueError(f"unsupported dim type: {dim_type}")

    def _ordered_names_for_dim_type(self, dim_type: isl.dim_type) -> tuple[str, ...]:
        names = self._names_for_dim_type(dim_type)
        return tuple(sorted(names, key=self._name_to_dim.__getitem__))

    def _ordered_name_chunks(self) -> dict[isl.dim_type, tuple[str, ...]]:
        return {
            isl.dim_type.set: self._ordered_names_for_dim_type(isl.dim_type.set),
            isl.dim_type.in_: self._ordered_names_for_dim_type(isl.dim_type.in_),
            isl.dim_type.param: self._ordered_names_for_dim_type(isl.dim_type.param),
        }

    def _empty_grouped_names(self) -> dict[isl.dim_type, list[str]]:
        return {
            isl.dim_type.set: [],
            isl.dim_type.in_: [],
            isl.dim_type.param: [],
        }

    def _metadata_from_chunk_names(
        self, chunk_names: Mapping[isl.dim_type, Collection[str]], *, has_inputs: bool
    ) -> tuple[NameToDim, DimTypeToNames]:
        ordered_names = [
            *chunk_names[isl.dim_type.set],
            *chunk_names[isl.dim_type.in_],
            *chunk_names[isl.dim_type.param],
        ]
        new_name_to_dim: NameToDim = constantdict({
            name: dim for dim, name in enumerate(ordered_names)
        })

        new_dimtype_to_names: dict[isl.dim_type, frozenset[str]] = {}
        if has_inputs and chunk_names[isl.dim_type.in_]:
            new_dimtype_to_names[isl.dim_type.in_] = frozenset(
                chunk_names[isl.dim_type.in_]
            )
        if chunk_names[isl.dim_type.param]:
            new_dimtype_to_names[isl.dim_type.param] = frozenset(
                chunk_names[isl.dim_type.param]
            )

        return new_name_to_dim, constantdict(new_dimtype_to_names)

    def _add_names_by_dim_type(
        self, names_to_add: Collection[str], dim_type: isl.dim_type
    ) -> Self:
        if isinstance(self._obj, isl.PwMultiAff):
            raise NotImplementedError

        dim_type = _normalize_public_dim_type(dim_type)
        if dim_type not in (isl.dim_type.set, isl.dim_type.in_, isl.dim_type.param):
            raise ValueError(f"unsupported dim type: {dim_type}")
        if (
            not isinstance(self._obj, IslSetLike | IslMultiExpressionLike)
            and dim_type == isl.dim_type.set
        ):
            raise ValueError(f"unsupported dim type: {dim_type}")

        if len(set(names_to_add)) != len(tuple(names_to_add)):
            raise ValueError("duplicate names to add")

        for name in names_to_add:
            if name in self.names:
                raise ValueError(f"name already exists: {name}")

        if not names_to_add:
            return self

        grouped_names = self._empty_grouped_names()
        grouped_names[dim_type] = list(names_to_add)

        return self._add_grouped_names(grouped_names)

    def _add_grouped_names(
        self, grouped_names: Mapping[isl.dim_type, Collection[str]]
    ) -> Self:
        if isinstance(self._obj, isl.PwMultiAff):
            raise NotImplementedError

        seen_names: set[str] = set()
        for names in grouped_names.values():
            for name in names:
                if name in seen_names:
                    raise ValueError("duplicate names to add")
                if name in self.names:
                    raise ValueError(f"name already exists: {name}")
                seen_names.add(name)

        new_obj = self._obj
        chunk_names = {
            dt: list(names) for dt, names in self._ordered_name_chunks().items()
        }
        internal_dim_type = (
            isl.dim_type.set
            if isinstance(new_obj, IslSetLike | IslMultiExpressionLike)
            else isl.dim_type.in_
        )

        insertion_starts = {
            isl.dim_type.set: 0,
            isl.dim_type.in_: len(chunk_names[isl.dim_type.set]),
            isl.dim_type.param: (
                len(chunk_names[isl.dim_type.set]) + len(chunk_names[isl.dim_type.in_])
            ),
        }

        for dim_type in (isl.dim_type.param, isl.dim_type.in_, isl.dim_type.set):
            names_to_add = grouped_names[dim_type]
            if not names_to_add:
                continue
            new_obj = new_obj.insert_dims(
                internal_dim_type, insertion_starts[dim_type], len(names_to_add)
            )
            chunk_names[dim_type] = [*names_to_add, *chunk_names[dim_type]]

        new_name_to_dim, new_dimtype_to_names = self._metadata_from_chunk_names(
            chunk_names,
            has_inputs=isinstance(new_obj, IslSetLike | IslMultiExpressionLike),
        )

        return type(self)(
            cast("IslObjectT", _restore_names(new_obj, new_name_to_dim)),
            new_name_to_dim,
            new_dimtype_to_names,
        )

    def add_names(self, tagged_names_to_add: Sequence[_TaggedName]) -> Self:
        grouped_names = self._empty_grouped_names()
        for tagged_name in tagged_names_to_add:
            dim_type = _normalize_public_dim_type(tagged_name._isl_dim_type)
            if dim_type not in grouped_names:
                raise ValueError(f"unsupported dim type: {tagged_name._isl_dim_type}")
            grouped_names[dim_type].append(tagged_name.name)

        return self._add_grouped_names(grouped_names)

    def add_set_names(self, names_to_add: Collection[str]) -> Self:
        return self._add_names_by_dim_type(names_to_add, isl.dim_type.set)

    def add_output_names(self, names_to_add: Collection[str]) -> Self:
        return self._add_names_by_dim_type(names_to_add, isl.dim_type.out)

    def add_input_names(self, names_to_add: Collection[str]) -> Self:
        return self._add_names_by_dim_type(names_to_add, isl.dim_type.in_)

    def add_parameter_names(self, names_to_add: Collection[str]) -> Self:
        return self._add_names_by_dim_type(names_to_add, isl.dim_type.param)

    def add_dim_names(
        self, names_to_add: Collection[str], dim_type: isl.dim_type
    ) -> Self:
        return self._add_names_by_dim_type(names_to_add, dim_type)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._name_to_dim.keys())

    def dim_names(self, dim_type: isl.dim_type) -> frozenset[str]:
        return self._names_for_dim_type(dim_type)

    def ordered_dim_names(self, dim_type: isl.dim_type) -> tuple[str, ...]:
        return self._ordered_names_for_dim_type(dim_type)

    @property
    def set_names(self) -> frozenset[str]:
        return self._names_for_dim_type(isl.dim_type.set)

    @property
    def output_names(self) -> frozenset[str]:
        return self._names_for_dim_type(isl.dim_type.out)

    def get_space(self) -> isl.Space:
        return self._reconstruct_isl_object().get_space()

    def dim(self, dim_type: isl.dim_type) -> int:
        dim_type = _normalize_public_dim_type(dim_type)
        if dim_type in (isl.dim_type.set, isl.dim_type.in_, isl.dim_type.param):
            return len(self._names_for_dim_type(dim_type))
        return self._reconstruct_isl_object().dim(dim_type)

    def move_dims(
        self,
        names_to_move: str | Collection[str],
        dst_type: isl.dim_type,
    ) -> Self:
        if isinstance(names_to_move, str):
            names_to_move = [names_to_move]

        if not names_to_move:
            return self

        dst_type = _normalize_public_dim_type(dst_type)
        if dst_type not in (isl.dim_type.set, isl.dim_type.in_, isl.dim_type.param):
            raise ValueError(f"unsupported destination dim type: {dst_type}")

        missing_names = [name for name in names_to_move if name not in self.names]
        if missing_names:
            raise ValueError(f"unknown names: {', '.join(missing_names)}")

        if len(set(names_to_move)) != len(tuple(names_to_move)):
            raise ValueError("duplicate names in move_dims")

        names_to_move = [
            name
            for name in names_to_move
            if name not in self._names_for_dim_type(dst_type)
        ]
        if not names_to_move:
            return self

        moved_name_set = set(names_to_move)
        chunk_names = {
            dt: [name for name in names if name not in moved_name_set]
            for dt, names in self._ordered_name_chunks().items()
        }
        moved_names = sorted(names_to_move, key=self._name_to_dim.__getitem__)
        chunk_names[dst_type].extend(moved_names)

        new_name_to_dim, new_dimtype_to_names = self._metadata_from_chunk_names(
            chunk_names,
            has_inputs=True,
        )

        return _align_obj(self, new_name_to_dim, new_dimtype_to_names)

    def rename_dims(self, renaming: Mapping[str, str]) -> Self:
        if not renaming:
            return self

        missing_names = [name for name in renaming if name not in self.names]
        if missing_names:
            raise ValueError(f"unknown names: {', '.join(missing_names)}")

        if len(set(renaming.values())) != len(renaming):
            raise ValueError("duplicate destination names in rename_dims")

        unchanged_names = {
            old_name for old_name, new_name in renaming.items() if old_name == new_name
        }
        renaming = {
            old_name: new_name
            for old_name, new_name in renaming.items()
            if old_name not in unchanged_names
        }
        if not renaming:
            return self

        existing_names = self.names - frozenset(renaming)
        conflicting_names = existing_names & frozenset(renaming.values())
        if conflicting_names:
            raise ValueError(
                "cannot rename to existing names: "
                + ", ".join(sorted(conflicting_names))
            )

        new_name_to_dim: NameToDim = constantdict({
            renaming.get(name, name): dim for name, dim in self._name_to_dim.items()
        })
        new_dimtype_to_names: DimTypeToNames = constantdict({
            dim_type: frozenset(renaming.get(name, name) for name in names)
            for dim_type, names in self._dimtype_to_names.items()
        })

        return type(self)(self._obj, new_name_to_dim, new_dimtype_to_names)

    @overload
    def equate_dims(self, name1: Mapping[str, str]) -> Self: ...

    @overload
    def equate_dims(self, name1: str, name2: str) -> Self: ...

    def equate_dims(
        self,
        name1: str | Mapping[str, str],
        name2: str | None = None,
    ) -> Self:
        if isinstance(name1, str):
            if name2 is None:
                raise TypeError("name2 must be provided when name1 is a string")
            equated_names = ((name1, name2),)
        else:
            if name2 is not None:
                raise TypeError("name2 cannot be provided when name1 is a mapping")
            equated_names = tuple(name1.items())

        for lhs_name, rhs_name in equated_names:
            if lhs_name not in self.names:
                raise ValueError(f"unknown name: {lhs_name}")
            if rhs_name not in self.names:
                raise ValueError(f"unknown name: {rhs_name}")

        if all(lhs_name == rhs_name for lhs_name, rhs_name in equated_names):
            return self

        if not isinstance(self._obj, IslSetLike):
            raise NotImplementedError(
                "equate_dims is only implemented for set-like objects"
            )

        obj = self._obj
        for lhs_name, rhs_name in equated_names:
            if lhs_name != rhs_name:
                obj = obj.equate(
                    isl.dim_type.set,
                    self._name_to_dim[lhs_name],
                    isl.dim_type.set,
                    self._name_to_dim[rhs_name],
                )

        return type(self)(
            cast("IslObjectT", obj),
            self._name_to_dim,
            self._dimtype_to_names,
        )

    @property
    def _has_inputs(self) -> bool:
        return bool(self._metadata_input_names)

    @property
    def input_names(self) -> frozenset[str]:
        return self._names_for_dim_type(isl.dim_type.in_)

    @property
    def _input_dim_start(self) -> int | None:
        if self._has_inputs:
            return min(self._name_to_dim[name] for name in self._metadata_input_names)
        return None

    @property
    def _has_params(self) -> bool:
        return bool(self._metadata_parameter_names)

    @property
    def parameter_names(self) -> frozenset[str]:
        return self._metadata_parameter_names

    @property
    def _parameter_dim_start(self) -> int | None:
        if self._has_params:
            return min(
                self._name_to_dim[name] for name in self._metadata_parameter_names
            )
        return None

    def _reconstruct_isl_object(self) -> IslObject:
        """
        Relies on the dimension type ordering in
        :func:`_deconstruct_set_like_object`.
        """
        obj = cast(
            "IslSetLike | IslBaseExpressionLike | IslPwExpressionLike | isl.MultiAff",
            _restore_names(self._obj, self._name_to_dim),
        )

        internal_dim = (
            isl.dim_type.set if isinstance(obj, isl.Set) else isl.dim_type.in_
        )

        if self._has_params:
            if self._parameter_dim_start is None:
                raise ValueError(
                    "Object has parameter dimensions, but a starting index for "
                    "parameter names is not given. Reconstruction is not "
                    "possible"
                )

            param_start = self._parameter_dim_start
            obj = obj.move_dims(
                isl.dim_type.param,
                0,
                internal_dim,
                param_start,
                len(self.parameter_names),
            )

        if self._has_inputs:
            if self._input_dim_start is None:
                raise ValueError(
                    "Object has input dimensions, but a starting index for "
                    "input names is not given. Reconstruction is not "
                    "possible"
                )

            obj_domain = isl.Set("{ [] }")
            obj_range = obj
            assert isinstance(obj_range, isl.BasicSet | isl.Set)

            obj = isl.Map.from_domain_and_range(obj_domain, obj_range)

            inp_start = self._input_dim_start
            obj = obj.move_dims(
                isl.dim_type.in_, 0, internal_dim, inp_start, len(self.input_names)
            )

        return obj

    def get_isl_object(self) -> IslObject:
        return self._reconstruct_isl_object()

    @override
    def __str__(self) -> str:
        return str(self._reconstruct_isl_object())
