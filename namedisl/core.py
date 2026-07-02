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
from dataclasses import dataclass, field
from importlib import metadata
from typing import Generic, TypeAlias, TypeVar, cast, final, overload

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
IslScalarExpressionLike = IslBaseExpressionLike | IslPwExpressionLike
IslMultiExpressionLike = isl.MultiAff | isl.PwMultiAff

IslExpressionLike = IslScalarExpressionLike
IslSetLike = isl.BasicSet | isl.BasicMap | isl.Set | isl.Map
IslObject = IslSetLike | IslExpressionLike
RawInternalIslObject = IslSetLike | IslScalarExpressionLike
InternalIslObject = RawInternalIslObject

IslObjectT = TypeVar("IslObjectT", bound=IslObject)

T = TypeVar("T")

IslExpressionLikeT = TypeVar(
    "IslExpressionLikeT",
    bound=IslScalarExpressionLike,
)
RawInternalIslObjectT_co = TypeVar(
    "RawInternalIslObjectT_co",
    bound=RawInternalIslObject,
    covariant=True,
)
InternalIslObjectT_co = TypeVar(
    "InternalIslObjectT_co",
    bound=InternalIslObject,
    covariant=True,
)
PublicIslObjectT_co = TypeVar(
    "PublicIslObjectT_co",
    bound=IslObject,
    covariant=True,
)

NamedIslObjectT = TypeVar(
    "NamedIslObjectT",
    bound="NamedIslObject[InternalIslObject, IslObject]",
)
NamedIslObjectT2 = TypeVar(
    "NamedIslObjectT2",
    bound="NamedIslObject[InternalIslObject, IslObject]",
)


class DimId(NamedTuple):
    dim_type: DimType
    dim_index: int


NameToDim: TypeAlias = Mapping[str, DimId]

# NOTE: without tracking what dimension type a particular name belongs to, it is
# not possible to reconstruct the ISL object after dimension operations, e.g.
# alignment
DimTypeToNames: TypeAlias = Mapping[DimType, Sequence[str]]


__version__ = metadata.version("namedisl")
_match = re.match(r"^([0-9.]+)([a-z0-9]*?)$", __version__)
assert _match
VERSION = tuple(int(nr) for nr in _match.group(1).split("."))

__all__ = [
    "_align_and_apply_binary_op",
    "_align_two",
    "_deconstruct_object",
    "_find_contiguous_dim_chunks",
    # "_make_named_object_pieces",
    # "_normalize_dimtype_to_names",
    "_restore_names",
    "_strip_names",
]


def not_none(obj: T | None) -> T:
    assert obj is not None
    return obj


def _uses_explicit_input_metadata(obj: object) -> bool:
    return isinstance(obj, IslSetLike)


def _strip_names(
    obj: RawInternalIslObjectT_co,
) -> tuple[RawInternalIslObjectT_co, NameToDim]:
    name_to_dim: dict[str, int] = {}

    dt_to_strip = DimType.set if isinstance(obj, IslSetLike) else DimType.in_

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

    return cast("RawInternalIslObjectT_co", stripped_obj), constantdict(name_to_dim)


# def _get_obj_dim_name(obj: IslObject, dt: DimType, dim: int) -> str:
#     if isinstance(obj, isl.QPolynomial | isl.PwQPolynomial):
#         name = obj.space.get_dim_name(dt, dim)
#     else:
#         name = obj.get_dim_name(dt, dim)

#     if name is None:
#         raise ValueError("unnamed dimension found")

#     return name


# def _normalize_dimtype_to_names(
#     obj: IslObject, dimtype_to_names: DimTypeToNames
# ) -> DimTypeToNames:
#     if isinstance(obj, IslSetLike):
#         dim_type = dim_type.set
#         total_dims = obj.dim(dim_type)
#         n_in = len(dimtype_to_names.get(dim_type.in_, frozenset()))
#         n_param = len(dimtype_to_names.get(dim_type.param, frozenset()))

#         new_dimtype_to_names: dict[DimType, frozenset[str]] = {}

#         if n_in:
#             start = total_dims - n_param - n_in
#             new_dimtype_to_names[dim_type.in_] = frozenset(
#                 _get_obj_dim_name(obj, dim_type, dim)
#                 for dim in range(start, start + n_in)
#             )

#         if n_param:
#             start = total_dims - n_param
#             new_dimtype_to_names[dim_type.param] = frozenset(
#                 _get_obj_dim_name(obj, dim_type, dim)
#                 for dim in range(start, start + n_param)
#             )

#         return constantdict(new_dimtype_to_names)

#     total_dims = obj.dim(dim_type.in_)
#     n_param = len(dimtype_to_names.get(dim_type.param, frozenset()))
#     if not n_param:
#         return dimtype_to_names

#     start = total_dims - n_param
#     return constantdict({
#         dim_type.param: frozenset(
#             _get_obj_dim_name(obj, dim_type.in_, dim)
#             for dim in range(start, start + n_param)
#         )
#     })


def _dimtype_to_names(obj: IslObject) -> DimTypeToNames:
    space = obj.space
    return {
        dt: [
            not_none(space.get_dim_name(dt.value, i))
            for i in range(obj.dim(dt.as_isl()))]
        for dt in DimType}


def _restore_names(
    obj: IslObjectT, name_to_dim: NameToDim
) -> IslObjectT:
    """
    Return a copy of *obj* with dimension names restored from *name_to_dim*.

    This is intentionally used at reconstruction boundaries.  Internal
    metadata-only operations may leave the private isl object's own dimension
    names stale because namedisl does not rely on those names for correctness.
    """
    restored_obj = obj
    if isinstance(restored_obj, isl.PwAff):
        # input dimensions cannot be renamed for isl.PwAff, so we first move
        # input dims to the parameter space, rename then move back
        restored_obj = restored_obj.move_dims(
            DimType.param,
            0,
            DimType.in_,
            0,
            restored_obj.dim(DimType.in_),
        )

        restored_union_pw_aff = restored_obj.to_union_pw_aff()
        for name, dim in name_to_dim.items():
            restored_union_pw_aff = restored_union_pw_aff.set_dim_name(
                DimType.param, dim, name
            )

        restored_obj = restored_union_pw_aff.get_pw_aff_list().get_at(0)
        return cast(
            "RawInternalIslObjectT_co",
            restored_obj.move_dims(
                DimType.in_,
                0,
                DimType.param,
                0,
                restored_obj.dim(DimType.param),
            ),
        )

    if isinstance(restored_obj, IslSetLike):
        dt_to_restore = DimType.set
    else:
        dt_to_restore = DimType.in_

    for name, dim in name_to_dim.items():
        restored_obj = restored_obj.set_dim_name(dt_to_restore, dim, name)

    if isinstance(restored_obj, isl.UnionPwAff | isl.UnionPwMultiAff):
        raise NotImplementedError

    return cast("RawInternalIslObjectT_co", restored_obj)


def _get_dim_names(obj: IslObject, dt: DimType) -> frozenset[str]:
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
    obj1: NamedIslObject[InternalIslObject, IslObject],
    obj2: NamedIslObject[InternalIslObject, IslObject],
) -> DimTypeToNames:
    """
    Enforces alphabetical ordering of all dimensions found in :arg:`obj1` and
    :arg:`obj2` within each dimension-type chunk. This ordering is used in
    alignment before performing operations between two set-like objects.
    """
    all_set_names = {*obj1.set_names, *obj2.set_names}
    all_inp_names = {*obj1.input_names, *obj2.input_names}
    all_param_names = {*obj1.parameter_names, *obj2.parameter_names}

    duplicate_names = (
        (all_set_names & all_inp_names)
        | (all_set_names & all_param_names)
        | (all_inp_names & all_param_names)
    )
    if duplicate_names:
        raise ValueError(
            "duplicate dimension names across dimension types: "
            + ", ".join(sorted(duplicate_names))
        )

    return constantdict({
        DimType.param: sorted(all_param_names),
        DimType.in_: sorted(all_inp_names),
        DimType.set: sorted(all_set_names),
    })


def _align_obj(
    named_obj: NamedIslObjectT,
    dimtype_to_names: DimTypeToNames,
    *, allow_cross_dim_type: bool = False,
) -> NamedIslObjectT:
    """
    Return *named_obj* with internal dimensions arranged according to
    *dimtype_to_names*.

    The isl object is moved or expanded as needed, but public names are carried
    in metadata and are restored only when a raw isl object is reconstructed.
    """
    new_isl_obj = named_obj._obj
    running_name_to_dim_id = dict(named_obj._name_to_dim)

    for target_dt, names in dimtype_to_names.items():
        for target_dim, name in enumerate(names):
            old_dim_id = running_name_to_dim_id.get(name)
            target_dim_id = DimId(target_dt, target_dim)
            if old_dim_id is not None:
                if old_dim_id == target_dim_id:
                    continue

                if old_dim_id.dim_type == target_dim_id.dim_type:
                    another_dim_type = DimType.param
                    if another_dim_type == old_dim_id.dim_type:
                        another_dim_type = DimType.set

                    new_isl_obj = new_isl_obj.move_dims(
                        another_dim_type.as_isl(), 0,
                        old_dim_id.dim_type.as_isl(), old_dim_id.dim_index,
                        1)
                    new_isl_obj = new_isl_obj.move_dims(
                        target_dim_id.dim_type.as_isl(), target_dim_id.dim_index,
                        another_dim_type.as_isl(), 0,
                        1)

                else:
                    if not allow_cross_dim_type:
                        raise ValueError("moves across dim_types are not allowed")

                    new_isl_obj = new_isl_obj.move_dims(
                        target_dim_id.dim_type.as_isl(), target_dim_id.dim_index,
                        old_dim_id.dim_type.as_isl(), old_dim_id.dim_index,
                        1)

            else:
                old_dim_id = DimId(target_dt, new_isl_obj.dim(target_dt.as_isl()))
                new_isl_obj = new_isl_obj.insert_dims(target_dt.as_isl(), target_dim, 1)

            # track side effects of inserting/swapping dimensions

            old_dim = old_dim_id.dim_index
            for n, (dt, d) in list(running_name_to_dim_id.items()):
                if (target_dim > old_dim) and (d > old_dim):
                    running_name_to_dim_id[n] = DimId(dt, d - 1)
                elif (target_dim < old_dim) and (d < old_dim):
                    running_name_to_dim_id[n] = DimId(dt, d + 1)

            running_name_to_dim_id[name] = DimId(target_dt, target_dim)

    return type(named_obj)(
        new_isl_obj,
        _dimtype_to_names=dimtype_to_names,
    )


def _align_two(
    named_obj1: NamedIslObjectT, named_obj2: NamedIslObjectT2
) -> tuple[NamedIslObjectT, NamedIslObjectT2]:
    """
    Align two named isl objects to a common name-to-dimension mapping.
    """

    dimtype_to_names = _find_joint_name_to_dim(named_obj1, named_obj2)

    named_obj1 = _align_obj(named_obj1, dimtype_to_names)
    named_obj2 = _align_obj(named_obj2, dimtype_to_names)

    return named_obj1, named_obj2


def _align_and_apply_binary_op(
    lhs: NamedIslObject[InternalIslObjectT_co, IslObject],
    rhs: NamedIslObject[InternalIslObjectT_co, IslObject],
    op: Callable[
        [InternalIslObjectT_co, InternalIslObjectT_co], InternalIslObjectT_co
    ],
) -> NamedIslObject[InternalIslObjectT_co, IslObject]:
    """
    Align *lhs* and *rhs*, apply *op* to their isl objects, and wrap the result.
    """

    lhs, rhs = _align_two(lhs, rhs)
    result = op(lhs._obj, rhs._obj)
    return type(lhs)(result, _dimtype_to_names=lhs._dimtype_to_names)


@dataclass(frozen=True, eq=False)
class NamedIslObject(ABC, Generic[InternalIslObjectT_co, PublicIslObjectT_co]):
    """
    Base class for named isl wrappers.

    Instances pair a private isl object with metadata that records the semantic
    name and public dimension kind of every internal dimension.  Subclasses use
    this metadata to implement operations in terms of names while still
    delegating the underlying integer-set algebra to isl.
    """

    _obj: InternalIslObjectT_co

    _dimtype_to_names: DimTypeToNames = field(kw_only=True)

    if __debug__:
        def __post_init__(self):
            all_names: list[str] = []
            for names in self._dimtype_to_names.values():
                all_names.extend(names)
            if len(all_names) != len(set(all_names)):
                raise ValueError("names must be unique across dim types")

            assert DimType.param in self._dimtype_to_names
            assert DimType.in_ in self._dimtype_to_names
            assert DimType.set in self._dimtype_to_names

    @property
    def _name_to_dim(self) -> NameToDim:
        try:
            return self._name_to_dim_cache  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        except KeyError:
            pass

        result = {name: DimId(dt, i)
            for dt, names in self._dimtype_to_names.items()
            for i, name in enumerate(names)}

        object.__setattr__(self, "_name_to_dim_cache", result)
        return result

    def add_dim_names(
        self, names_to_add: Collection[str], dt: DimType
    ) -> Self:
        all_names = [*names_to_add, *self.names]
        if len(set(all_names)) != len(all_names):
            raise ValueError("duplicate names after addition")

        new_dimtype_to_names = {
            **self._dimtype_to_names,
            dt: [*self._dimtype_to_names[dt], *names_to_add]
        }
        obj = cast("InternalIslObjectT_co",
            self._obj.insert_dims(isl.dim_type(dt), self.dim(dt), len(names_to_add)))

        return type(self)(obj, _dimtype_to_names=new_dimtype_to_names)

    @property
    def names(self) -> Collection[str]:
        return self._name_to_dim.keys()

    @property
    def param_names(self) -> Collection[str]:
        return self._dimtype_to_names[DimType.param]

    @property
    def in_names(self) -> Collection[str]:
        return self._dimtype_to_names[DimType.in_]

    @property
    def set_names(self) -> Collection[str]:
        return self._dimtype_to_names[DimType.set]

    def _get_space(self) -> isl.Space:
        """
        Reconstruct and return the object's public isl space.
        """
        return self._reconstruct_isl_object().get_space()

    def dim(self, dim_type: DimType) -> int:
        return len(self._dimtype_to_names[dim_type])

    def move_dims(
        self,
        names_to_move: Collection[str],
        dst_type: DimType,
    ) -> Self:
        """
        Return a copy with named dimensions moved to *dst_type*.

        The relative order of moved names is preserved.  Moving a name to its
        current dimension kind is a no-op.
        """
        if isinstance(names_to_move, str):
            raise TypeError("names_to_move must be a collection")

        if not names_to_move:
            return self

        missing_names = [name for name in names_to_move if name not in self.names]
        if missing_names:
            raise ValueError(f"unknown names: {', '.join(missing_names)}")

        if len(set(names_to_move)) != len(tuple(names_to_move)):
            raise ValueError("duplicate names in move_dims")

        source_dim_types = {self.dt for name in names_to_move}

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
            has_inputs=_uses_explicit_input_metadata(self._obj),
        )

        return _align_obj(self, new_name_to_dim, new_dimtype_to_names)

    def rename_dims(self, renaming: Mapping[str, str]) -> Self:
        """
        Return a copy with dimension names changed according to *renaming*.
        """
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

        return type(self)(
            self._obj,
            new_name_to_dim,
            new_dimtype_to_names,
        )

    @property
    def _has_inputs(self) -> bool:
        return bool(self._metadata_input_names)

    @property
    def input_names(self) -> Collection[str]:
        """
        Names of input dimensions.
        """
        return self._names_for_dim_type(DimType.in_)

    @property
    def _input_dim_start(self) -> int | None:
        if self._has_inputs:
            return min(self._name_to_dim[name] for name in self._metadata_input_names)
        return None

    @property
    def _has_params(self) -> bool:
        return bool(self._metadata_parameter_names)

    @property
    def parameter_names(self) -> Collection[str]:
        """
        Names of parameter dimensions.
        """
        return self._metadata_parameter_names

    @property
    def _parameter_dim_start(self) -> int | None:
        if self._has_params:
            return min(
                self._name_to_dim[name] for name in self._metadata_parameter_names
            )
        return None

    def _reconstruct_isl_object(self) -> PublicIslObjectT_co:
        """
        Relies on the dimension type ordering in
        :func:`_deconstruct_set_like_object`.
        """
        obj = _restore_names(self._obj, self._name_to_dim)

        internal_dim = (
            DimType.set if isinstance(obj, isl.Set) else DimType.in_
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
                DimType.param,
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
                DimType.in_, 0, internal_dim, inp_start, len(self.input_names)
            )

        return cast("PublicIslObjectT_co", obj)

    def get_isl_object(self) -> PublicIslObjectT_co:
        """
        Reconstruct and return the wrapped public :mod:`islpy` object.
        """
        return self._reconstruct_isl_object()

    @override
    def __str__(self) -> str:
        return str(self._reconstruct_isl_object())
