"""
.. autoclass:: Set
.. autoclass:: Map

.. autofunction:: make_set
.. autofunction:: make_map
"""


from __future__ import annotations
from warnings import warn


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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Generic, Self, TypeAlias, TypeVar, overload

from constantdict import constantdict

import islpy as isl
from islpy import dim_type


__version__ = metadata.version("namedisl")
_match = re.match(r"^([0-9.]+)([a-z0-9]*?)$", __version__)
assert _match
VERSION = tuple(int(nr) for nr in _match.group(1).split("."))

ALL_DIM_TYPES = [dim_type.param, dim_type.set, dim_type.in_, dim_type.div]


IslSetLikeObject = isl.Set | isl.Map

IslExpressionLikeObject = isl.Aff | isl.QPolynomial
IslPwExpressionLikeObject = isl.PwAff | isl.PwQPolynomial
IslMultiExpressionLikeObject = isl.MultiAff | isl.PwMultiAff

IslAllExpressionLike = (
    IslExpressionLikeObject |
    IslPwExpressionLikeObject |
    IslMultiExpressionLikeObject
)

IslObject = IslSetLikeObject | IslAllExpressionLike
NameToDim: TypeAlias = Mapping[str, tuple[isl.dim_type, int]]

IslTypeT = TypeVar("IslTypeT", bound=IslObject)


@dataclass(frozen=True)
class NamedIslObject(Generic[IslTypeT]):
    _obj: IslTypeT
    _name_to_dim: NameToDim
    # TODO: add cache

    # FIXME: better name?
    def dim_type_names(
        self,
        dim_type: dim_type
    ) -> Sequence[str]:
        return [
            name
            for name, (dt, _) in self._name_to_dim.items()
            if dim_type == dt
        ]

    def dim(self, name: str) -> int:
        dt, _ = self._name_to_dim[name]
        return self._obj.dim(dt)



NamedIslTypeT = TypeVar("NamedIslTypeT", bound=NamedIslObject[Any])


# {{{ utils

def _strip_names(obj: IslTypeT) -> tuple[IslTypeT, NameToDim]:
    name_to_dim = {}
    for tp in ALL_DIM_TYPES:
        for i in range(obj.dim(tp)):
            # NOTE: not all ExpressionLike objects have `get_dim_name` as an
            # available method
            name = obj.get_space().get_dim_name(tp, i)

            if name is None and not isinstance(obj, IslAllExpressionLike):
                raise ValueError("unnamed dimension found")
            if name in name_to_dim and name is not None:
                raise ValueError(f"non-unique dim name: {name}")

            # FIXME: allow unnamed (None) dimensions for ExpressionLike objects
            # for now
            if name is None and isinstance(obj, IslExpressionLikeObject):
                warn("Unnamed dimension found in expression-like object.")

            name_to_dim[name] = (tp, i)

    return obj, constantdict(name_to_dim)


def _restore_names(named_obj: NamedIslObject) -> IslObject:
    obj = named_obj._obj
    name_to_dim = named_obj._name_to_dim

    for name, (dt, i) in name_to_dim.items():
        obj = obj.set_dim_name(dt, i, name)

    # FIXME: disallow these until needed
    if isinstance(obj, isl.UnionPwAff | isl.UnionPwMultiAff):
        raise NotImplementedError()

    return obj


def _find_joint_name_to_dim(obj: NameToDim, template: NameToDim) -> NameToDim:
    """
    Uses `template` to determine a name-to-dimension mapping used for aligning
    the spaces between `obj` and `template`.
    """
    name_to_dim = dict(template)

    shared_names = set(name_to_dim.keys()) & set(obj.keys())
    for name in sorted(shared_names):
        dim_type_obj, _ = obj[name]
        dim_type_template, _ = template[name]
        if dim_type_obj != dim_type_template:
            raise ValueError(
                f"{name} belongs to a different dim_type in `obj` than in "
                "`template`"
            )

    dim_type_to_idx = dict.fromkeys(ALL_DIM_TYPES, -1)
    for dim_type, pos in template.values():
        dim_type_to_idx[dim_type] = max(dim_type_to_idx[dim_type], pos + 1)

    unique_names = set(obj.keys()) - shared_names
    for name in sorted(unique_names):
        dim_type, _ = obj[name]
        pos = dim_type_to_idx[dim_type]
        name_to_dim = constantdict(dict(name_to_dim) | {name: (dim_type, pos)})

        dim_type_to_idx[dim_type] += 1

    return name_to_dim


def _align_space(obj: NamedIslTypeT, ordering: NameToDim) -> NamedIslTypeT:
    """
    Aligns the space and name-to-dimension mapping of `obj` to match what is
    specified by `ordering`. Returns a new object whose dims are aligned
    according to `ordering`.
    """

    if isinstance(obj._obj, isl.MultiAff | isl.PwMultiAff):
        raise ValueError(
        "Must align each component of a MultiExpression individually")

    new_isl_obj = obj._obj.copy()
    new_name_to_dim = dict(obj._name_to_dim)

    for name, (dim_type, pos) in sorted(ordering.items(), key=lambda k: k[1][1]):

        if name in obj._name_to_dim:
            _, old_pos = new_name_to_dim[name]
            if old_pos == pos:
                    continue

            temp_dim_type = isl.dim_type.param
            if temp_dim_type == dim_type:
                temp_dim_type = isl.dim_type.set

            temp_pos = new_isl_obj.dim(temp_dim_type)

            new_isl_obj = new_isl_obj.move_dims(
                temp_dim_type, temp_pos, dim_type, old_pos, 1)
            new_isl_obj = new_isl_obj.move_dims(
                dim_type, pos, temp_dim_type, temp_pos, 1)
        else:
            old_pos = new_isl_obj.dim(dim_type)
            new_isl_obj = new_isl_obj.insert_dims(dim_type, pos, 1)

            if dim_type == isl.dim_type.param:
                new_isl_obj = new_isl_obj.set_dim_name(dim_type, pos, name)

        temp_name_to_dim = new_name_to_dim.copy()
        for cur_name, (cur_dt, cur_pos) in sorted(new_name_to_dim.items(),
                                                  key=lambda k: k[1][1]):
            if cur_dt != dim_type:
                continue

            if (pos > old_pos) and (cur_pos > old_pos):
                temp_name_to_dim[cur_name] = (cur_dt, cur_pos - 1)
            elif (pos < old_pos) and (cur_pos < old_pos):
                temp_name_to_dim[cur_name] = (cur_dt, cur_pos + 1)

        # FIXME: renaming dimensions can recast PwAff as UnionPwAff. leaving
        # this as-is until needed
        if isinstance(new_isl_obj, isl.UnionPwAff):
            raise NotImplementedError("UnionPwAff unsupported")

        new_name_to_dim = temp_name_to_dim
        new_name_to_dim[name] = (dim_type, pos)

    return type(obj)(new_isl_obj, ordering)


def _align_two(obj1: NamedIslTypeT,
               obj2: NamedIslTypeT) -> Sequence[NamedIslTypeT]:
    """
    Aligns the spaces and name-to-dimension mappings of `obj1` and `obj2` so
    that they are compatible for *named* set operations. `obj2` will first be
    aligned to `obj1`, then `obj1` will be aligned to the result of the first
    alignment.
    """
    if type(obj1) != type(obj2):
        raise ValueError(
            "Alignment requires both objects are of the same type. Got "
           f"type(obj1) = {type(obj1)} and type(obj2) = {type(obj2)}"
        )

    ordering = _find_joint_name_to_dim(obj2._name_to_dim, obj1._name_to_dim)

    obj2 = _align_space(obj2, ordering)
    obj1 = _align_space(obj1, ordering)

    return obj1, obj2


def _align_and_apply_op(
        obj1: NamedIslTypeT,
        obj2: NamedIslTypeT,
        op: Callable[[NamedIslTypeT, NamedIslTypeT], IslTypeT]
    ) -> NamedIslTypeT:
    if type(obj1) != type(obj2):
        raise ValueError(
            "Operations between objects requires both objects are of the same "
           f"type. Got type(obj1) = {type(obj1)} and type(obj2) = {type(obj2)}"
        )

    obj1, obj2 = _align_two(obj1, obj2)
    result = op(obj1._obj, obj2._obj)

    return type(obj1)(result, obj1._name_to_dim)

# }}}


# {{{ set-like objects

@dataclass(frozen=True)
class _SetLike(NamedIslObject):
    _obj: IslSetLikeObject

    def add_dims(self, name_to_dim_type: Mapping[str, dim_type]) -> Self:
        name_to_dim = dict(self._name_to_dim)
        obj = self._obj

        for name, dt in name_to_dim_type.items():
            ndims = self._obj.dim(dt)

            # NOTE: names in ISL are "None" when added, so give it a name
            obj = self._obj.insert_dims(dt, ndims, 1)
            obj = obj.set_dim_name(dt, ndims, name)

            name_to_dim = name_to_dim | {name : (dt, ndims)}

        return type(self)(obj, name_to_dim)

    def rename_dims(self, old_name_to_new_name: Mapping[str, str]) -> Self:
        name_to_dim = dict(self._name_to_dim)
        obj = self._obj

        for old_name, new_name in old_name_to_new_name.items():
            dt, dim_pos = self._name_to_dim[old_name]
            obj = self._obj.set_dim_name(dt, dim_pos, new_name)
            name_to_dim = name_to_dim | {new_name : (dt, dim_pos)}

            del name_to_dim[old_name]

        return type(self)(obj, name_to_dim)

    def complement(self) -> Self:
        return type(self)(self._obj.complement(), self._name_to_dim)

    def dim_max(self, name) -> PwAff:
        _, pos = self._name_to_dim[name]

        return make_pw_aff(self._obj.dim_max(pos))

    def dim_min(self, name) -> PwAff:
        _, pos = self._name_to_dim[name]

        return make_pw_aff(self._obj.dim_min(pos))

    def as_pw_multi_aff(self) -> PwMultiAff:
        return make_pw_multi_aff(self._obj.as_pw_multi_aff())

    def eliminate(self, names: str | Sequence[str]) -> Self:
        if isinstance(names, str):
            names = [names]

        new_obj = self._obj
        for name in names:
            dt, pos = self._name_to_dim[name]
            new_obj = new_obj.eliminate(dt, pos, 1)

        return type(self)(new_obj, self._name_to_dim)

    def project_out(self, names_to_project_out: str | Sequence[str]) -> Self:
        if isinstance(names_to_project_out, str):
            names_to_project_out = [names_to_project_out]

        new_name_to_dim = dict(self._name_to_dim)
        new_isl_obj = self._obj
        for proj_name in names_to_project_out:
            dt, pos = new_name_to_dim[proj_name]
            new_name_to_dim.pop(proj_name)

            temp_name_to_dim = new_name_to_dim.copy()
            for name, (cur_dt, cur_pos) in sorted(
                    new_name_to_dim.items(), key=lambda k: k[1][1]):
                if cur_pos > pos:
                    temp_name_to_dim[name] = (cur_dt, cur_pos - 1)

            new_name_to_dim = temp_name_to_dim.copy()

            new_isl_obj = new_isl_obj.project_out(dt, pos, 1)

        return type(self)(new_isl_obj, new_name_to_dim)

    def project_out_except(self, names_to_keep: str | Sequence[str]) -> Self:
        if isinstance(names_to_keep, str):
            names_to_keep = [names_to_keep]

        names_to_project_out = [
            name for name in self._name_to_dim.keys()
            if name not in names_to_keep
        ]

        return self.project_out(names_to_project_out)

    def __and__(self, other) -> Self:
        return _align_and_apply_op(self, other, operator.and_)

    def __eq__(self, other) -> bool:
        aligned_self, aligned_other = _align_two(self, other)
        return aligned_self._obj.plain_is_equal(aligned_other._obj)  # type: ignore

    def __or__(self, other) -> Self:
        return _align_and_apply_op(self, other, operator.or_)

    def __sub__(self, other) -> Self:
        return _align_and_apply_op(self, other, operator.sub)

    def __str__(self) -> str:
        return str(_restore_names(self))


@dataclass(frozen=True, eq=False)
class Set(_SetLike):
    _obj: isl.Set


@overload
def make_set(src: str, ctx: isl.Context | None = None) -> Set:
    ...


@overload
def make_set(src: isl.Set | isl.BasicSet) -> Set:
    ...


def make_set(
        src: str | isl.Set | isl.BasicSet,
        ctx: isl.Context | None = None
    ) -> Set:
    if isinstance(src, str):
        obj = isl.Set(src, ctx)
    elif isinstance(src, isl.BasicSet):
        obj = isl.Set.from_basic_set(src)
    else:
        obj = src

    obj, name_to_dim = _strip_names(obj)
    return Set(obj, name_to_dim)


@dataclass(frozen=True, eq=False)
class Map(_SetLike):
    _obj: isl.Map

    def domain(self) -> Set:
        return make_set(self._obj.domain())

    def range(self) -> Set:
        return make_set(self._obj.range())

    def reverse(self) -> Map:
        return make_map(self._obj.reverse())


@overload
def make_map(src: str, ctx: isl.Context | None = None) -> Map:
    ...


@overload
def make_map(src: isl.Map | isl.BasicMap) -> Map:
    ...


def make_map(src: str | isl.Map | isl.BasicMap,
             ctx: isl.Context | None = None) -> Map:
    if isinstance(src, str):
        obj = isl.Map(src, ctx)
    elif isinstance(src, isl.BasicMap):
        obj = isl.Map.from_basic_map(src)
    else:
        obj = src

    obj, name_to_dim = _strip_names(obj)
    return Map(obj, name_to_dim)

# }}}


# {{{ expression-like objects

@dataclass(frozen=True)
class _ExpressionLike(NamedIslObject):
    _obj: IslExpressionLikeObject

    def get_constant_val(self) -> isl.Val:
        return self._obj.get_constant_val()



@dataclass(frozen=True, eq=False)
class Aff(_ExpressionLike):
    _obj: isl.Aff

    def get_coefficient_val(self, name: str) -> isl.Val:
        dt, pos = self._name_to_dim[name]
        return self._obj.get_coefficient_val(dt, pos)

    def get_denominator_val(self) -> isl.Val:
        return self._obj.get_denominator_val()

    def get_div(self, name) -> Aff:
        _, pos = self._name_to_dim[name]
        return make_aff(self._obj.get_div(pos))


@overload
def make_aff(src: str, ctx: isl.Context | None = None) -> Aff:
    ...


@overload
def make_aff(src: isl.Aff) -> Aff:
    ...


def make_aff(src: str | isl.Aff, ctx: isl.Context | None = None) -> Aff:
    obj = isl.Aff(src, ctx) if isinstance(src, str) else src

    obj, name_to_dim = _strip_names(obj)
    return Aff(obj, name_to_dim)


@dataclass(frozen=True, eq=False)
class QPolynomial(_ExpressionLike):
    """
    No constructor exists for QPolynomials, so we do not implement any "make"
    methods for QPolynomials.
    """
    _obj: isl.QPolynomial


def make_qpolynomial(src: isl.QPolynomial) -> QPolynomial:
    obj, name_to_dim = _strip_names(src)
    return QPolynomial(obj, name_to_dim)

# }}}


# {{{ multi expressions

@dataclass(frozen=True)
class _MultiExpressionLike(NamedIslObject):
    _obj: isl.MultiAff | isl.PwMultiAff

    def get_at(self, dim_name: str) -> PwAff | Aff:
        _, dim_pos = self._name_to_dim[dim_name]
        obj_at = self._obj.get_at(dim_pos)

        if isinstance(obj_at, isl.PwAff):
            return make_pw_aff(obj_at)
        else:
            return make_aff(obj_at)


@dataclass(frozen=True, eq=False)
class MultiAff(_MultiExpressionLike):
    _obj: isl.MultiAff


@overload
def make_multi_aff(src: str, ctx: isl.Context | None = None) -> MultiAff:
    ...


@overload
def make_multi_aff(src: isl.MultiAff) -> MultiAff:
    ...


def make_multi_aff(src: str | isl.MultiAff,
                   ctx: isl.Context | None = None) -> MultiAff:
    obj = isl.MultiAff(src, ctx) if isinstance(src, str) else src

    obj, name_to_dim = _strip_names(obj)
    return MultiAff(obj, name_to_dim)


@dataclass(frozen=True, eq=False)
class PwMultiAff(_MultiExpressionLike):
    _obj: isl.PwMultiAff


@overload
def make_pw_multi_aff(src: str, ctx: isl.Context | None = None) -> PwMultiAff:
    ...


@overload
def make_pw_multi_aff(src: isl.PwMultiAff) -> PwMultiAff:
    ...


def make_pw_multi_aff(src: str | isl.PwMultiAff,
                      ctx: isl.Context | None = None) -> PwMultiAff:
    obj = isl.PwMultiAff(src, ctx) if isinstance(src, str) else src
    obj, name_to_dim = _strip_names(obj)

    return PwMultiAff(obj, name_to_dim)

# }}}


# {{{ piecewise expressions

@dataclass(frozen=True)
class _PwExpressionLike(NamedIslObject):
    _obj: isl.PwAff | isl.PwQPolynomial

    def get_pieces(self) -> Sequence:
        named_pieces = []
        for (dom, expn) in self._obj.get_pieces():

            named_dom = make_set(dom)
            if isinstance(expn, isl.Aff):
                named_expn = make_aff(expn)
            else:
                named_expn = make_qpolynomial(expn)

            named_pieces.append((named_dom, named_expn))

        return named_pieces


@dataclass(frozen=True, eq=False)
class PwAff(_PwExpressionLike):
    _obj: isl.PwAff


@overload
def make_pw_aff(src: str, ctx: isl.Context | None = None) -> PwAff:
    ...


@overload
def make_pw_aff(src: isl.PwAff) -> PwAff:
    ...


def make_pw_aff(src: str | isl.PwAff, ctx: isl.Context | None = None) -> PwAff:
    obj = isl.PwAff(src, ctx) if isinstance(src, str) else src

    obj, name_to_dim = _strip_names(obj)
    return PwAff(obj, name_to_dim)


@dataclass(frozen=True, eq=False)
class PwQPolynomial(_PwExpressionLike):
    _obj: isl.PwQPolynomial


@overload
def make_pw_qpolynomial(src: str,
                        ctx: isl.Context | None = None) -> PwQPolynomial:
    ...


@overload
def make_pw_qpolynomial(src: isl.PwQPolynomial) -> PwQPolynomial:
    ...


def make_pw_qpolynomial(src: str | isl.PwQPolynomial,
                        ctx: isl.Context | None = None) -> PwQPolynomial:
    obj = isl.PwQPolynomial(src, ctx) if isinstance(src, str) else src

    obj, name_to_dim = _strip_names(obj)
    return PwQPolynomial(obj, name_to_dim)


# }}}
