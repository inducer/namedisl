"""
Core metadata machinery for name-aware isl wrappers.

The public set-like and expression-like classes store an isl object together
with metadata that records which semantic name belongs to each internal
dimension.  This module
contains the alignment, reconstruction, and metadata-manipulation helpers used
to keep that invariant intact.
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

import enum
import re
from abc import ABC
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from importlib import metadata
from typing import ClassVar, Generic, TypeAlias, TypeVar, cast, final

from constantdict import constantdict
from typing_extensions import NamedTuple, Self, override

import islpy as isl


@final
class DimType(enum.IntEnum):
    param = isl.dim_type.param
    in_ = isl.dim_type.in_
    set = isl.dim_type.set

    def as_isl(self):
        return isl.dim_type(self)


IslBaseExpressionLike = isl.Aff | isl.QPolynomial
IslPwExpressionLike = isl.PwAff | isl.PwQPolynomial
IslExpressionLike = IslBaseExpressionLike | IslPwExpressionLike
IslMultiExpressionLike = isl.MultiAff | isl.PwMultiAff

IslSetLike = isl.BasicSet | isl.Set
IslMapLike = isl.BasicMap | isl.Map
IslSetOrMapLike = IslSetLike | IslMapLike
IslObject = IslSetOrMapLike | IslExpressionLike | IslMultiExpressionLike

IslObjectT = TypeVar("IslObjectT", bound=IslObject)
IslObjectT2 = TypeVar("IslObjectT2", bound=IslObject)
IslObjectT_co = TypeVar("IslObjectT_co", bound=IslObject, covariant=True)

T = TypeVar("T")

IslSetOrMapLikeT = TypeVar(
    "IslSetOrMapLikeT",
    bound=IslSetOrMapLike,
)
IslSetLikeT = TypeVar(
    "IslSetLikeT",
    bound=IslSetLike,
)
IslMapLikeT = TypeVar(
    "IslMapLikeT",
    bound=IslMapLike,
)
IslExpressionLikeT = TypeVar(
    "IslExpressionLikeT",
    bound=IslExpressionLike,
)
IslMultiExpressionLikeT = TypeVar(
    "IslMultiExpressionLikeT",
    bound=IslMultiExpressionLike,
)
IslMultiExpressionLikeT_co = TypeVar(
    "IslMultiExpressionLikeT_co",
    bound=IslMultiExpressionLike,
    covariant=True
)

NamedIslObjectT = TypeVar(
    "NamedIslObjectT",
    bound="NamedIslObject[IslObject]",
)
NamedIslObjectT2 = TypeVar(
    "NamedIslObjectT2",
    bound="NamedIslObject[IslObject]",
)


class DimId(NamedTuple):
    dim_type: DimType
    dim_index: int


NameToDim: TypeAlias = Mapping[str, DimId]
DimTypeToNames: TypeAlias = Mapping[DimType, Sequence[str]]


__version__ = metadata.version("namedisl")
_match = re.match(r"^([0-9.]+)([a-z0-9]*?)$", __version__)
assert _match
VERSION = tuple(int(nr) for nr in _match.group(1).split("."))

__all__ = [
    "_align_and_apply_binary_op",
    "_align_two",
    "_restore_names",
    "chunk_indices",
]


def not_none(obj: T | None) -> T:
    assert obj is not None
    return obj


def _dimtype_to_names(
            obj: IslObject,
            active_dim_types: Collection[DimType]
        ) -> DimTypeToNames:
    gdn_obj = (
        obj.space if isinstance(obj, (isl.QPolynomial, isl.PwQPolynomial)) else obj)

    return {
        dt: tuple(not_none(gdn_obj.get_dim_name(dt.value, i))
            for i in range(obj.dim(dt.as_isl())))
        for dt in active_dim_types}


def _restore_names(
    obj: IslObjectT, dimtype_to_names: DimTypeToNames,
) -> IslObjectT:
    for dt, names in dimtype_to_names.items():
        for name, dim in zip(names, range(obj.dim(dt.as_isl())), strict=True):
            obj = cast("IslObjectT", obj.set_dim_name(dt.as_isl(), dim, name))

    return obj


class IndexChunk(NamedTuple):
    start: int
    cnt: int


def chunk_indices(dims: Sequence[int]) -> list[IndexChunk]:
    assert dims == sorted(dims)

    chunks: list[IndexChunk] = []

    start = dims[0]
    count = 1

    from itertools import pairwise

    for prev, curr in pairwise(dims):
        if curr == prev + 1:
            count += 1
        else:
            chunks.append(IndexChunk(start, count))
            start = curr
            count = 1

    chunks.append(IndexChunk(start, count))

    return chunks


def chunked_dims_by_type(names: Collection[str], name_to_dim: NameToDim):
    """
    Chunks are guaranteed to be returned in ascending order
    """
    if isinstance(names, str):
        raise TypeError("expected collection of names, got string")

    dim_to_indices: dict[DimType, list[int]] = {}
    for name in names:
        source_dt, idx = name_to_dim[name]
        dim_to_indices.setdefault(source_dt, []).append(idx)

    return {dt: chunk_indices(sorted(idxs)) for dt, idxs in dim_to_indices.items()}


# FIXME: think through whether or not alphabetical ordering will require more
# work on average than using one of the objects as a template in alignment
def _find_joint_space(
    space1: Space,
    space2: Space,
) -> Space:
    """
    Enforces alphabetical ordering of all dimensions found in :arg:`obj1` and
    :arg:`obj2` within each dimension-type chunk. This ordering is used in
    alignment before performing operations between two set-like objects.
    """
    assert space1.dimtype_to_names.keys() == space2.dimtype_to_names.keys()

    dim_type_to_names = {
        dt: {
            *space1.dimtype_to_name_sets[dt],
            *space2.dimtype_to_name_sets[dt]
        }
        for dt in space1.dimtype_to_names
    }
    for dt1, names1 in dim_type_to_names.items():
        for dt2, names2 in dim_type_to_names.items():
            if dt1 >= dt2:
                continue
            dup_names = names1 & names2
            if dup_names:
                raise ValueError(
                    "duplicate dimension names across dimension types: "
                    f"{', '.join(sorted(dup_names))} across {dt1} and {dt2}"
                )

    return Space(constantdict({
        dt: tuple(sorted(names))
        for dt, names in dim_type_to_names.items()
    }))


def _align_obj(
    named_obj: NamedIslObjectT,
    space: Space,
    *, allow_cross_dim_type: bool = False,
) -> NamedIslObjectT:
    obj = named_obj._obj
    running_name_to_dim_id = dict(named_obj.sp.name_to_dim)

    if isinstance(obj, isl.PwMultiAff):
        raise NotImplementedError

    for target_dt, names in space.dimtype_to_names.items():
        for target_dim, name in enumerate(names):
            old_dim_id = running_name_to_dim_id.get(name)
            target_dim_id = DimId(target_dt, target_dim)

            if old_dim_id is None:
                obj = obj.insert_dims(target_dt.as_isl(), target_dim, 1)
                if target_dt == DimType.param:
                    # isl doesn't seem to like unnamed param dimensions. Make it happy.
                    obj = obj.set_dim_name(target_dt.as_isl(), target_dim, name)

                    # ban spooky islpy upcasts
                    assert not isinstance(obj, isl.UnionPwAff)

            else:
                if old_dim_id == target_dim_id:
                    continue

                if old_dim_id.dim_type == target_dim_id.dim_type:
                    another_dim_type = DimType.param
                    if another_dim_type == old_dim_id.dim_type:
                        another_dim_type = DimType.set

                    obj = obj.move_dims(
                        another_dim_type.as_isl(), 0,
                        old_dim_id.dim_type.as_isl(), old_dim_id.dim_index,
                        1)
                    obj = obj.move_dims(
                        target_dim_id.dim_type.as_isl(), target_dim_id.dim_index,
                        another_dim_type.as_isl(), 0,
                        1)

                else:
                    if not allow_cross_dim_type:
                        raise ValueError("moves across dim_types are not allowed")

                    obj = obj.move_dims(
                        target_dim_id.dim_type.as_isl(), target_dim_id.dim_index,
                        old_dim_id.dim_type.as_isl(), old_dim_id.dim_index,
                        1)

            # FIXME: This is kind of expensive.
            for other_name in list(running_name_to_dim_id):
                other_dt, other_idx = running_name_to_dim_id[other_name]
                if (old_dim_id
                        and other_dt == old_dim_id.dim_type
                        and other_idx > old_dim_id.dim_index):
                    other_idx -= 1
                if other_dt == target_dt and other_idx >= target_dim:
                    other_idx += 1
                running_name_to_dim_id[other_name] = DimId(other_dt, other_idx)

            running_name_to_dim_id[name] = DimId(target_dt, target_dim)

    return type(named_obj)(obj, space)


def _align_two(
    named_obj1: NamedIslObjectT, named_obj2: NamedIslObjectT2
) -> tuple[NamedIslObjectT, NamedIslObjectT2]:
    if named_obj1.sp.order_equal(named_obj2.sp):
        return named_obj1, named_obj2

    space = _find_joint_space(named_obj1.sp, named_obj2.sp)

    named_obj1 = _align_obj(named_obj1, space)
    named_obj2 = _align_obj(named_obj2, space)

    return named_obj1, named_obj2


def _align_for_compostition(
    lhs: NamedIslObject[IslObjectT],
    lhs_dt: DimType,
    rhs: NamedIslObject[IslObjectT2],
    rhs_dt: DimType,
) -> tuple[NamedIslObject[IslObjectT], NamedIslObject[IslObjectT2]]:
    interface_names_set = {
        *lhs.sp.dimtype_to_names[lhs_dt],
        *rhs.sp.dimtype_to_names[rhs_dt],
    }

    lhs_intersection = lhs.sp.names_except([lhs_dt]) & interface_names_set
    rhs_intersection = rhs.sp.names_except([rhs_dt]) & interface_names_set
    if lhs_intersection:
        raise ValueError(
            f"LHS names intersect with interface: {', '.join(lhs_intersection)}")
    if rhs_intersection:
        raise ValueError(
            f"RHS names intersect with interface: {', '.join(rhs_intersection)}")
    remaining_intersection = (
        lhs.sp.names_except([lhs_dt, DimType.param])
        & rhs.sp.names_except([rhs_dt, DimType.param])
    )
    if remaining_intersection:
        raise ValueError(
            f"Uninvolved names intersect : {', '.join(remaining_intersection)}")

    param_names = tuple(sorted({
        *lhs.sp.dimtype_to_names[DimType.param],
        *rhs.sp.dimtype_to_names[DimType.param],
    }))

    interface_names = tuple(sorted(interface_names_set))
    lhs_sp = Space(constantdict({
        **lhs.sp.dimtype_to_names,
        DimType.param: param_names,
        lhs_dt: interface_names,
    }))
    rhs_sp = Space(constantdict({
        **rhs.sp.dimtype_to_names,
        DimType.param: param_names,
        rhs_dt: interface_names,
    }))

    lhs = _align_obj(lhs, lhs_sp)
    rhs = _align_obj(rhs, rhs_sp)
    return lhs, rhs


def _align_and_apply_binary_op(
    lhs: NamedIslObject[IslObjectT],
    rhs: NamedIslObject[IslObjectT],
    op: Callable[
        [IslObjectT, IslObjectT], IslObjectT
    ],
) -> NamedIslObject[IslObject]:
    lhs, rhs = _align_two(lhs, rhs)
    result = op(lhs._obj, rhs._obj)
    return type(lhs)(result, lhs.sp)


@dataclass(frozen=True, eq=False)
class Space:
    dimtype_to_names: DimTypeToNames

    if __debug__:
        def __post_init__(self):
            hash(self.dimtype_to_names)

            all_names: list[str] = []
            for names in self.dimtype_to_names.values():
                all_names.extend(names)
            if len(all_names) != len(set(all_names)):
                raise ValueError("names must be unique across dim types")

    @staticmethod
    def from_names(
        param: Sequence[str] | None = None,
        in_: Sequence[str] | None = None,
        set: Sequence[str] | None = None,
    ):
        dim_type_to_names: dict[DimType, tuple[str, ...]] = {}
        if param is not None:
            dim_type_to_names[DimType.param] = tuple(param)
        if in_ is not None:
            dim_type_to_names[DimType.in_] = tuple(in_)
        if set is not None:
            dim_type_to_names[DimType.set] = tuple(set)
        return Space(constantdict(dim_type_to_names))

    @staticmethod
    def from_isl(obj: IslObject, dim_types: Collection[DimType]):
        return Space(constantdict(_dimtype_to_names(obj, dim_types)))

    @override
    def __eq__(self, other: object):
        raise RuntimeError("use .order_equal or .semantically_equal")

        # FIXME: Reenable for consistency with hash
        if not isinstance(other, Space):
            return False
        return self.order_equal(other)

    def order_equal(self, other: Space):
        if self is other:
            return True

        return self.dimtype_to_names == other.dimtype_to_names

    def semantically_equal(self, other: Space):
        if self is other:
            return True
        if not isinstance(other, Space):
            return False
        if hash(self) != hash(other):
            return False

        return self.dimtype_to_name_sets == other.dimtype_to_name_sets

    @override
    def __hash__(self):
        return hash((type(self), self.dimtype_to_names))

    @property
    def name_to_dim(self) -> NameToDim:
        try:
            return self._name_to_dim_cache  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        except AttributeError:
            pass

        result = {name: DimId(dt, i)
            for dt, names in self.dimtype_to_names.items()
            for i, name in enumerate(names)}

        object.__setattr__(self, "_name_to_dim_cache", result)
        return result

    @cached_property
    def dimtype_to_name_sets(self) -> Mapping[DimType, frozenset[str]]:
        return {
            dt: frozenset(names)
            for dt, names in self.dimtype_to_names.items()
        }

    @cached_property
    def names(self) -> frozenset[str]:
        return frozenset(self.name_to_dim.keys())

    def dim_names(self, dim_type: DimType) -> frozenset[str]:
        return self.dimtype_to_name_sets[dim_type]

    @property
    def param_names(self) -> frozenset[str]:
        return self.dimtype_to_name_sets[DimType.param]

    @property
    def in_names(self) -> frozenset[str]:
        return self.dimtype_to_name_sets[DimType.in_]

    @property
    def set_names(self) -> frozenset[str]:
        return self.dimtype_to_name_sets[DimType.set]

    def dim(self, dim_type: DimType) -> int:
        return len(self.dimtype_to_names[dim_type])

    def names_except(self, dim_type: Collection[DimType]):
        return {name
            for dt, names in self.dimtype_to_names.items()
            if dt not in dim_type
            for name in names}

    def move_dim_type(self, source: DimType, target: DimType) -> Space:
        if target in self.dimtype_to_names:
            raise ValueError(f"target dim type {target} already exists")
        new_dim_type_to_names = dict(self.dimtype_to_names)
        new_dim_type_to_names[target] = new_dim_type_to_names[source]
        del new_dim_type_to_names[source]
        return Space(constantdict(new_dim_type_to_names))

    def swap_dim_types(self, dt1: DimType, dt2: DimType) -> Space:
        new_dim_type_to_names = dict(self.dimtype_to_names)
        new_dim_type_to_names[dt1], new_dim_type_to_names[dt2] = \
            new_dim_type_to_names[dt2], new_dim_type_to_names[dt1]
        return Space(constantdict(new_dim_type_to_names))

    def drop_dim_type(self, dt: DimType) -> Space:
        new_dim_type_to_names = dict(self.dimtype_to_names)
        del new_dim_type_to_names[dt]
        return Space(constantdict(new_dim_type_to_names))

    def as_expr_space(self) -> Space:
        try:
            return self._expr_space_cache  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        except AttributeError:
            pass

        # In isl's exxpression-like spaces, "set" dimensions become "in" dimensions
        result = self.move_dim_type(DimType.set, DimType.in_)
        object.__setattr__(self, "_expr_space_cache", result)
        return result

    def as_set_space(self) -> Space:
        try:
            return self._set_space_cache  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        except AttributeError:
            pass

        # In isl's exxpression-like spaces, "set" dimensions become "in" dimensions
        result = self.move_dim_type(DimType.in_, DimType.set)
        object.__setattr__(self, "_set_space_cache", result)
        return result

    def as_isl(self, ctx: isl.Context | None = None):
        if ctx is None:
            ctx = isl.DEFAULT_CONTEXT
        result = isl.Space.alloc(
            ctx,
            nparam=len(self.dimtype_to_names.get(DimType.param, ())),
            n_in=len(self.dimtype_to_names.get(DimType.in_, ())),
            n_out=len(self.dimtype_to_names.get(DimType.set, ())),
        )

        for dim_type, names in self.dimtype_to_names.items():
            for i, name in enumerate(names):
                result = result.set_dim_name(dim_type.as_isl(), i, name)

        return result

    def as_isl_set_space(self, ctx: isl.Context | None = None):
        if self.dimtype_to_names.get(DimType.in_, []):
            raise ValueError("in-dimensions not allowed")

        if ctx is None:
            ctx = isl.DEFAULT_CONTEXT
        result = isl.Space.set_alloc(
            ctx,
            nparam=len(self.dimtype_to_names.get(DimType.param, ())),
            dim=len(self.dimtype_to_names.get(DimType.set, ())),
        )

        for dim_type, names in self.dimtype_to_names.items():
            for i, name in enumerate(names):
                result = result.set_dim_name(dim_type.as_isl(), i, name)

        return result


@dataclass(frozen=True, eq=False)
class NamedIslObject(ABC, Generic[IslObjectT_co]):
    # NOTE: _obj holds names, but they are not kept up to date and should not
    # be considered authoritative. See as_isl().
    _obj: IslObjectT_co

    sp: Space

    active_dim_types: ClassVar[frozenset[DimType]]

    if __debug__:
        def __post_init__(self):
            if frozenset(self.sp.dimtype_to_names) != self.active_dim_types:
                raise ValueError(
                    f"space not suitable for '{type(self)}'")

    def add_dim_names(
        self, dt: DimType, names_to_add: Collection[str], /
    ) -> Self:
        all_names = [*names_to_add, *self.sp.names]
        if len(set(all_names)) != len(all_names):
            raise ValueError("duplicate names after addition")

        new_dimtype_to_names = {
            **self.sp.dimtype_to_names,
            dt: (*self.sp.dimtype_to_names[dt], *names_to_add)
        }

        if isinstance(self._obj, isl.PwMultiAff):
            raise NotImplementedError

        obj = cast("IslObjectT_co",
            self._obj.insert_dims(dt.as_isl(), self.sp.dim(dt), len(names_to_add)))

        return type(self)(obj, Space(constantdict(new_dimtype_to_names)))

    def move_dims(
        self,
        names: Collection[str],
        dest_dt: DimType,
    ) -> Self:
        if not names:
            return self

        if isinstance(names, str):
            raise TypeError("names_to_move may not be a plain string")

        new_dimtype_to_names = {
            dt: list(names) for dt, names in self.sp.dimtype_to_names.items()}

        obj = self._obj

        for source_dt, chunks in chunked_dims_by_type(
                names, self.sp.name_to_dim).items():
            for start, count in chunks[::-1]:
                del new_dimtype_to_names[source_dt][start:start+count]
                new_dimtype_to_names[dest_dt].extend(self.sp.dimtype_to_names[source_dt][start:start+count])
                isl_dest_dt = dest_dt.as_isl()

                if isinstance(obj, isl.PwMultiAff):
                    raise NotImplementedError

                obj = cast("IslObjectT_co",
                    obj.move_dims(
                        isl_dest_dt, obj.dim(isl_dest_dt),
                        source_dt.as_isl(), start,
                        count))

        return type(self)(obj, Space(constantdict({
            dt: tuple(names)
            for dt, names in new_dimtype_to_names.items()
        })))

    def rename_dims(self, renaming: Mapping[str, str]) -> Self:
        if not renaming:
            return self

        new_dimtype_to_names = {
            dt: list(names) for dt, names in self.sp.dimtype_to_names.items()}

        for old_name, new_name in renaming.items():
            if new_name in renaming:
                raise ValueError(
                    "renaming may not map to a name to be renamed: "
                    f"'{new_name}'")
            if new_name in self.sp.names:
                raise ValueError(
                    f"cannot rename to existing name: '{new_name}'")

            dim_type, idx = self.sp.name_to_dim[old_name]
            new_dimtype_to_names[dim_type][idx] = new_name

        return type(self)(self._obj, Space(constantdict({
            dt: tuple(names)
            for dt, names in new_dimtype_to_names.items()
        })))

    def as_isl(self) -> IslObjectT_co:
        return _restore_names(self._obj, self.sp.dimtype_to_names)

    @override
    def __str__(self) -> str:
        return str(self.as_isl())
