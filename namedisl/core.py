"""
.. autofunction:: align_two

.. currentmodule:: namedisl
.. autoclass:: DimType
.. autoclass:: Space
.. autoclass:: Cache
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
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from importlib import metadata
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Concatenate,
    Generic,
    ParamSpec,
    TypeAlias,
    TypeVar,
    cast,
    final,
)

from constantdict import constantdict
from typing_extensions import NamedTuple, Self, override

import islpy as isl


if TYPE_CHECKING:
    from islpy._isl import dim_type


@final
class DimType(enum.IntEnum):
    """
    .. autoattribute:: param
    .. autoattribute:: in_
    .. autoattribute:: out

    .. automethod:: as_isl
    """
    param = isl.dim_type.param
    in_ = isl.dim_type.in_
    out = isl.dim_type.set

    def as_isl(self) -> dim_type:
        return isl.dim_type(self)


IslAffLike = isl.Aff | isl.PwAff
IslPolynomialLike = isl.QPolynomial | isl.PwQPolynomial
IslScalarExpressionLike = IslAffLike | IslPolynomialLike
IslMultiExpressionLike = isl.MultiAff | isl.PwMultiAff
IslExpressionLike = IslScalarExpressionLike | IslMultiExpressionLike

IslSetLike = isl.BasicSet | isl.Set
IslMapLike = isl.BasicMap | isl.Map
IslBasic = isl.BasicSet | isl.BasicMap
IslUnbasic = isl.Set | isl.Map
IslSetOrMapLike = IslSetLike | IslMapLike
IslObject = (
    IslSetOrMapLike | IslScalarExpressionLike | IslMultiExpressionLike | isl.Constraint)

IslObjectT = TypeVar("IslObjectT", bound=IslObject)
IslObjectT2 = TypeVar("IslObjectT2", bound=IslObject)
IslObjectT_co = TypeVar("IslObjectT_co", bound=IslObject, covariant=True)

T = TypeVar("T")
P  = ParamSpec("P")
R = TypeVar("R")

IslSetOrMapLikeT = TypeVar(
    "IslSetOrMapLikeT",
    bound=IslSetOrMapLike,
)
IslSetOrMapLikeT_co = TypeVar(
    "IslSetOrMapLikeT_co",
    bound=IslSetOrMapLike,
    covariant=True
)
IslSetLikeT = TypeVar(
    "IslSetLikeT",
    bound=IslSetLike,
)
IslMapLikeT = TypeVar(
    "IslMapLikeT",
    bound=IslMapLike,
)
IslBasicT_co = TypeVar(
    "IslBasicT_co",
    bound=IslBasic,
    covariant=True
)
IslUnbasicT_co = TypeVar(
    "IslUnbasicT_co",
    bound=IslUnbasic,
    covariant=True
)
IslScalarExpressionLikeT = TypeVar(
    "IslScalarExpressionLikeT",
    bound=IslScalarExpressionLike,
)
IslMultiExpressionLikeT_co = TypeVar(
    "IslMultiExpressionLikeT_co",
    bound=IslMultiExpressionLike,
    covariant=True
)
IslExpressionLikeT = TypeVar(
    "IslExpressionLikeT",
    bound=IslExpressionLike,
)
IslAffLikeT_co = TypeVar(
    "IslAffLikeT_co",
    bound=IslAffLike,
    covariant=True,
)
IslPolynomialLikeT_co = TypeVar(
    "IslPolynomialLikeT_co",
    bound=IslPolynomialLike,
    covariant=True,
)
IslExpressionLikeT_co = TypeVar(
    "IslExpressionLikeT_co",
    bound=IslExpressionLike,
    covariant=True,
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
    """
    .. autoattribute:: dim_type
    .. autoattribute:: dim_index
    """
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
    "_restore_names",
    "align_two",
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


def _set_dim_name(obj: IslObjectT, dt: DimType, idx: int, name: str) -> IslObjectT:
    # Ick, annoying. PwAff doesn't have native set_dim_name, but the Id's
    # are not considered equal to 'plain' names. As such, a situation
    # can arise where [n] -> ... and [n] -> ... will not be considered
    # equal, and arithmetic

    if isinstance(obj, isl.Constraint):
        raise NotImplementedError("setting names on Constraints")
    if isinstance(obj, (isl.PwAff, isl.PwMultiAff)):
        return cast("IslObjectT", obj.set_dim_id(dt.as_isl(), idx,
            isl.Id.read_from_str(obj.get_ctx(), name)))
    else:
        return cast("IslObjectT", obj.set_dim_name(dt.as_isl(), idx, name))


def _restore_names(
    obj: IslObjectT, dimtype_to_names: DimTypeToNames,
) -> IslObjectT:
    for dt, names in dimtype_to_names.items():
        if dt == DimType.param:
            # Those are kept up to date anyway because isl requires it.
            continue

        for name, dim in zip(names, range(obj.dim(dt.as_isl())), strict=True):
            obj = _set_dim_name(obj, dt, dim, name)

    return obj


class IndexChunk(NamedTuple):
    """
    .. autoattribute:: start
    .. autoattribute:: cnt
    """
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


def chunked_dims_by_type(
    names: Collection[str], name_to_dim: NameToDim
) -> dict[DimType, list[IndexChunk]]:
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


def align_obj(
    named_obj: NamedIslObjectT,
    space: Space,
    *, allow_cross_dim_type: bool = False,
) -> NamedIslObjectT:
    obj = named_obj._obj
    running_name_to_dim_id = dict(named_obj.space.name_to_dim)

    if isinstance(obj, (isl.PwMultiAff, isl.Constraint)):
        raise NotImplementedError

    for target_dt, names in space.dimtype_to_names.items():
        for target_dim, name in enumerate(names):
            old_dim_id = running_name_to_dim_id.get(name)
            target_dim_id = DimId(target_dt, target_dim)

            if old_dim_id is None:
                obj = obj.insert_dims(target_dt.as_isl(), target_dim, 1)
                if target_dt == DimType.param:
                    # isl doesn't seem to like unnamed param dimensions. Make it happy.
                    obj = _set_dim_name(obj, target_dt, target_dim, name)

                    # ban spooky islpy upcasts
                    assert not isinstance(obj, isl.UnionPwAff)

            else:
                if old_dim_id == target_dim_id:
                    continue

                if old_dim_id.dim_type == target_dim_id.dim_type:
                    another_dim_type = DimType.param
                    if another_dim_type == old_dim_id.dim_type:
                        # determine a safe 'alternate' dim type
                        if isinstance(obj, (isl.Set, isl.BasicSet)):
                            another_dim_type = DimType.out
                        else:
                            another_dim_type = DimType.in_

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


def align_two(
    named_obj1: NamedIslObjectT, named_obj2: NamedIslObjectT2
) -> tuple[NamedIslObjectT, NamedIslObjectT2]:
    if named_obj1.space.order_equals(named_obj2.space):
        return named_obj1, named_obj2

    space = _find_joint_space(named_obj1.space, named_obj2.space)

    named_obj1 = align_obj(named_obj1, space)
    named_obj2 = align_obj(named_obj2, space)

    return named_obj1, named_obj2


def align_expr_and_set(
    expr_obj: NamedIslObjectT, set_obj: NamedIslObjectT2,
) -> tuple[NamedIslObjectT, NamedIslObjectT2]:
    space = _find_joint_space(set_obj.space, expr_obj.space.as_set_space())

    set_obj = align_obj(set_obj, space)
    expr_obj = align_obj(expr_obj, space.as_expr_space())

    return expr_obj, set_obj


def align_for_compostition(
    lhs: NamedIslObject[IslObjectT],
    lhs_dt: DimType,
    rhs: NamedIslObject[IslObjectT2],
    rhs_dt: DimType,
) -> tuple[NamedIslObject[IslObjectT], NamedIslObject[IslObjectT2]]:
    interface_names_set = {
        *lhs.space.dimtype_to_names[lhs_dt],
        *rhs.space.dimtype_to_names[rhs_dt],
    }

    lhs_intersection = lhs.space.names_except([lhs_dt]) & interface_names_set
    rhs_intersection = rhs.space.names_except([rhs_dt]) & interface_names_set
    if lhs_intersection:
        raise ValueError(
            f"LHS names intersect with interface: {', '.join(lhs_intersection)}")
    if rhs_intersection:
        raise ValueError(
            f"RHS names intersect with interface: {', '.join(rhs_intersection)}")
    remaining_intersection = (
        lhs.space.names_except([lhs_dt, DimType.param])
        & rhs.space.names_except([rhs_dt, DimType.param])
    )
    if remaining_intersection:
        raise ValueError(
            f"Uninvolved names intersect : {', '.join(remaining_intersection)}")

    param_names = tuple(sorted({
        *lhs.space.dimtype_to_names[DimType.param],
        *rhs.space.dimtype_to_names[DimType.param],
    }))

    interface_names = tuple(sorted(interface_names_set))
    lhs_sp = Space(constantdict({
        **lhs.space.dimtype_to_names,
        DimType.param: param_names,
        lhs_dt: interface_names,
    }))
    rhs_sp = Space(constantdict({
        **rhs.space.dimtype_to_names,
        DimType.param: param_names,
        rhs_dt: interface_names,
    }))

    lhs = align_obj(lhs, lhs_sp)
    rhs = align_obj(rhs, rhs_sp)
    return lhs, rhs


def plain_is_equal(lhs: IslObjectT, rhs: IslObjectT) -> bool:
    # Expose the cheapest/strictest equality test for all isl object types.
    if isinstance(lhs, (isl.Set, isl.Map, isl.PwAff, isl.Aff, isl.MultiAff)):
        return lhs.plain_is_equal(rhs)
    if isinstance(lhs, (isl.BasicSet, isl.BasicMap, isl.Constraint)):
        # these don't have plain_is_equal
        return lhs.is_equal(rhs)

    elif isinstance(lhs, (isl.QPolynomial, isl.PwQPolynomial)):
        return (lhs - rhs).is_zero()
    elif isinstance(lhs, isl.PwMultiAff):
        return lhs.is_equal(rhs)
    else:
        raise NotImplementedError()


def _align_and_apply_binary_op(
    lhs: NamedIslObject[IslObjectT],
    rhs: NamedIslObject[IslObjectT],
    op: Callable[
        [IslObjectT, IslObjectT], IslObjectT
    ],
) -> NamedIslObject[IslObject]:
    lhs, rhs = align_two(lhs, rhs)
    result = op(lhs._obj, rhs._obj)
    return type(lhs)(result, lhs.space)


@dataclass(frozen=True, eq=False)
class Space:
    """
    .. autoattribute:: dimtype_to_names
    .. automethod:: from_names
    .. automethod:: from_isl
    .. automethod:: __eq__
    .. automethod:: __hash__
    .. automethod:: order_equals
    .. automethod:: semantically_equals
    .. automethod:: __contains__
    .. autoattribute:: name_to_dim
    .. autoattribute:: dimtype_to_name_sets
    .. autoattribute:: names
    .. automethod:: dim_names
    .. autoattribute:: param_names
    .. autoattribute:: in_names
    .. autoattribute:: set_names
    .. autoattribute:: out_names
    .. automethod:: dim
    .. automethod:: names_except
    .. automethod:: move_dim_type
    .. automethod:: swap_dim_types
    .. automethod:: drop_dim_type
    .. automethod:: with_empty_dim_type
    .. automethod:: as_expr_space
    .. automethod:: as_set_space
    .. automethod:: as_isl
    .. automethod:: as_isl_set_space
    """
    dimtype_to_names: DimTypeToNames

    if __debug__:
        def __post_init__(self) -> None:
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
    ) -> Space:
        dim_type_to_names: dict[DimType, tuple[str, ...]] = {}
        if param is not None:
            dim_type_to_names[DimType.param] = tuple(param)
        if in_ is not None:
            dim_type_to_names[DimType.in_] = tuple(in_)
        if set is not None:
            dim_type_to_names[DimType.out] = tuple(set)
        return Space(constantdict(dim_type_to_names))

    @staticmethod
    def from_isl(obj: IslObject, dim_types: Collection[DimType]) -> Space:
        return Space(constantdict(_dimtype_to_names(obj, dim_types)))

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Space):
            return False
        # - consistent with hash
        # - the strictest/cheapest possible check
        return self.order_equals(other)

    @override
    def __hash__(self) -> int:
        return hash((type(self), self.dimtype_to_names))

    def order_equals(self, other: Space) -> bool:
        if self is other:
            return True

        return self.dimtype_to_names == other.dimtype_to_names

    def semantically_equals(self, other: Space) -> bool:
        if self is other:
            return True
        if not isinstance(other, Space):
            return False
        if hash(self) != hash(other):
            return False

        return self.dimtype_to_name_sets == other.dimtype_to_name_sets

    def __contains__(self, name: str) -> bool:
        return name in self.name_to_dim

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
        return self.dimtype_to_name_sets[DimType.out]

    @property
    def out_names(self) -> frozenset[str]:
        return self.dimtype_to_name_sets[DimType.out]

    def dim(self, dim_type: DimType) -> int:
        return len(self.dimtype_to_names[dim_type])

    def names_except(self, dim_type: Collection[DimType]) -> set[str]:
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

    def with_empty_dim_type(self, dt: DimType) -> Space:
        new_dim_type_to_names = dict(self.dimtype_to_names)
        if dt in new_dim_type_to_names:
            raise ValueError(f"dim type {dt} already exists")
        new_dim_type_to_names[dt] = ()
        return Space(constantdict(new_dim_type_to_names))

    def as_expr_space(self) -> Space:
        try:
            return self._expr_space_cache  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        except AttributeError:
            pass

        # In isl's exxpression-like spaces, "set" dimensions become "in" dimensions
        result = self.move_dim_type(DimType.out, DimType.in_)
        object.__setattr__(self, "_expr_space_cache", result)
        return result

    def as_set_space(self) -> Space:
        try:
            return self._set_space_cache  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        except AttributeError:
            pass

        # In isl's exxpression-like spaces, "set" dimensions become "in" dimensions
        result = self.move_dim_type(DimType.in_, DimType.out)
        object.__setattr__(self, "_set_space_cache", result)
        return result

    def as_isl(self, ctx: isl.Context | None = None) -> isl.Space:
        if ctx is None:
            ctx = isl.DEFAULT_CONTEXT
        result = isl.Space.alloc(
            ctx,
            nparam=len(self.dimtype_to_names.get(DimType.param, ())),
            n_in=len(self.dimtype_to_names.get(DimType.in_, ())),
            n_out=len(self.dimtype_to_names.get(DimType.out, ())),
        )

        for dim_type, names in self.dimtype_to_names.items():
            for i, name in enumerate(names):
                result = result.set_dim_name(dim_type.as_isl(), i, name)

        return result

    def as_isl_set_space(self, ctx: isl.Context | None = None) -> isl.Space:
        if self.dimtype_to_names.get(DimType.in_, []):
            raise ValueError("in-dimensions not allowed")

        if ctx is None:
            ctx = isl.DEFAULT_CONTEXT
        result = isl.Space.set_alloc(
            ctx,
            nparam=len(self.dimtype_to_names.get(DimType.param, ())),
            dim=len(self.dimtype_to_names.get(DimType.out, ())),
        )

        for dim_type, names in self.dimtype_to_names.items():
            for i, name in enumerate(names):
                result = result.set_dim_name(dim_type.as_isl(), i, name)

        return result


@dataclass(frozen=True, eq=False)
class NamedIslObject(Generic[IslObjectT_co]):
    # NB: Assigning to __doc__ is goofy, but it allows for compatibility with
    # the docstring pasting scheme used in subclasses.
    __doc__ = """
    .. autoattribute:: _obj
    .. autoattribute:: space
    .. autoattribute:: active_dim_types
    .. automethod:: add_dims
    .. automethod:: move_dims
    .. automethod:: rename_dims
    .. automethod:: as_isl
    .. automethod:: __hash__
    .. automethod:: __eq__
    .. automethod:: __str__
    """
    # NOTE: _obj holds names, but they are not kept up to date and should not
    # be considered authoritative. See as_isl().
    _obj: IslObjectT_co
    space: Space
    _isl_names_ok: bool = False

    active_dim_types: ClassVar[frozenset[DimType]]

    if __debug__:
        def __post_init__(self) -> None:
            space = self.space
            if frozenset(space.dimtype_to_names) != self.active_dim_types:
                raise ValueError(
                    f"space not suitable for '{type(self)}'")
            isl_space = self._obj.space
            for dt in self.active_dim_types:
                if isl_space.dim(dt.as_isl()) != space.dim(dt):
                    raise ValueError(f"space dimensions for {dt} don't match")

    def add_dims(
        self, dt: DimType, names_to_add: Collection[str], /
    ) -> Self:
        if isinstance(names_to_add, str):
            raise TypeError("names_to_add may not be a plain string")

        all_names = [*names_to_add, *self.space.names]
        if len(set(all_names)) != len(all_names):
            raise ValueError("duplicate names after addition")

        new_dimtype_to_names = {
            **self.space.dimtype_to_names,
            dt: (*self.space.dimtype_to_names[dt], *names_to_add)
        }

        if isinstance(self._obj, (isl.PwMultiAff, isl.Constraint)):
            raise NotImplementedError

        start_dim = self.space.dim(dt)
        obj = cast("IslObjectT_co",
                   self._obj.insert_dims(dt.as_isl(), self.space.dim(dt),
                                         len(names_to_add)))

        if dt == DimType.param:
            # isl doesn't like unnamed param dimensions. Make it happy.
            for idx, name in enumerate(names_to_add):
                obj = _set_dim_name(obj, dt, start_dim+idx, name)

        return type(self)(obj,
                          Space(constantdict(new_dimtype_to_names)))

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
            dt: list(names) for dt, names in self.space.dimtype_to_names.items()}

        obj = self._obj

        for source_dt, chunks in chunked_dims_by_type(
                names, self.space.name_to_dim).items():
            for start, count in chunks[::-1]:
                del new_dimtype_to_names[source_dt][start:start+count]
                new_dimtype_to_names[dest_dt].extend(self.space.dimtype_to_names[source_dt][start:start+count])
                isl_dest_dt = dest_dt.as_isl()

                if isinstance(obj, (isl.PwMultiAff, isl.Constraint)):
                    raise NotImplementedError

                if dest_dt == DimType.param:
                    for offset, name in enumerate(
                            self.space.dimtype_to_names[source_dt][start:start+count]):
                        obj = _set_dim_name(obj, source_dt, start+offset, name)

                obj = cast("IslObjectT_co",
                    obj.move_dims(
                        isl_dest_dt, obj.dim(isl_dest_dt),
                        source_dt.as_isl(), start,
                        count))

        return type(self)(obj, Space(constantdict({
            dt: tuple(names)
            for dt, names in new_dimtype_to_names.items()
        })))

    def rename_dims(self, renaming: Iterable[tuple[str, str]]) -> Self:
        if not renaming:
            return self

        new_dimtype_to_names = {
            dt: list(names) for dt, names in self.space.dimtype_to_names.items()}

        obj = self._obj
        for old_name, new_name in renaming:
            if new_name in self.space.names:
                raise ValueError(
                    f"cannot rename to existing name: '{new_name}'")

            dim_type, idx = self.space.name_to_dim[old_name]

            # isl doesn't like unnamed param dimensions. Make it happy.
            obj = _set_dim_name(obj, dim_type, idx, new_name)

            new_dimtype_to_names[dim_type][idx] = new_name

        return type(self)(
            obj,
            Space(constantdict({
                dt: tuple(names)
                for dt, names in new_dimtype_to_names.items()
            })))

    def as_isl(self) -> IslObjectT_co:
        if self._isl_names_ok:
            return self._obj

        res = _restore_names(self._obj, self.space.dimtype_to_names)
        object.__setattr__(self, "_obj", res)
        object.__setattr__(self, "_isl_names_ok", True)
        return res

    @override
    def __hash__(self) -> int:
        return hash(self._obj)

    @override
    def __eq__(self, other: object) -> bool:
        """This is intended to be cheap and strict, suitable mainly for hashing.
        Some subclasses (e.g. :class:`namedisl.Set`) may provide a different,
        'mathematically exact' notion of equality as a different method
        that is more expensive to check for.
        """
        if type(self) is not type(other):
            return NotImplemented
        other = cast("Self", other)

        if not self.space.order_equals(other.space):
            return False

        return plain_is_equal(self._obj, other._obj)

    @override
    def __str__(self) -> str:
        return str(self.as_isl())


class Cache:
    _cache: dict[Hashable, list[tuple[IslObject, object]]]

    def __init__(self) -> None:
        self._cache = {}


def with_cache(
    cache: Cache | None,
    f: Callable[Concatenate[IslObjectT, P], R],
    obj: IslObjectT,
    *args: P.args,
    **kwargs: P.kwargs
) -> R:
    if cache is None:
        return f(obj, *args, **kwargs)

    # This is so complicated because islpy's __eq__ doesn't route to plain_is_equal,
    # but may instead use expensive forms of equality.
    obj_hash = hash(obj)
    key = (f, obj_hash, tuple(args), constantdict(kwargs))
    try:
        candidates = cast("list[tuple[IslObjectT, R]]", cache._cache[key])
    except KeyError:
        pass
    else:
        for cand_obj, result in candidates:
            if plain_is_equal(obj, cand_obj):
                return result

    result = f(obj, *args, **kwargs)
    cast(
        "list[tuple[IslObjectT, R]]",
        cache._cache.setdefault(key, [])
    ).append((obj, result))
    return result
