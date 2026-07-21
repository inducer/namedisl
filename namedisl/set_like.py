"""
.. currentmodule:: namedisl

Constraint
^^^^^^^^^^
.. autoclass:: Constraint
.. autofunction:: make_constraint

Quasiconvex set
^^^^^^^^^^^^^^^
.. autoclass:: BasicSet
.. autofunction:: make_basic_set

General set
^^^^^^^^^^^
.. autoclass:: Set
.. autofunction:: make_set

Quasiconvex map
^^^^^^^^^^^^^^^
.. autoclass:: BasicMap
.. autofunction:: make_basic_map

General map
^^^^^^^^^^^
.. autoclass:: Map
.. autofunction:: make_map
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
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Literal, cast, overload

from constantdict import constantdict
from typing_extensions import Self

import islpy as isl

from .core import (
    Cache,
    DimType,
    IslBasicT_co,
    IslMapLikeT,
    IslSetLikeT,
    IslSetOrMapLikeT,
    IslSetOrMapLikeT_co,
    IslUnbasicT_co,
    NamedIslObject,
    Space,
    _align_and_apply_binary_op,
    align_for_compostition,
    align_two,
    chunked_dims_by_type,
    with_cache,
)
from .expression_like import PwAff, make_pw_multi_aff


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from namedisl import Aff, PwMultiAff


def _compare_set_or_map_like(
    lhs: _NamedIslSetOrMapLike[IslSetOrMapLikeT],
    rhs: _NamedIslSetOrMapLike[IslSetOrMapLikeT],
    op: Callable[[IslSetOrMapLikeT, IslSetOrMapLikeT], bool],
) -> bool:
    if type(lhs) is not type(rhs):
        return NotImplemented

    aligned_lhs, aligned_rhs = align_two(lhs, rhs)

    return op(aligned_lhs._obj, aligned_rhs._obj)


class Constraint(NamedIslObject[isl.Constraint]):
    """
    .. automethod:: equality_from_aff
    .. automethod:: inequality_from_aff
    .. autoattribute:: is_equality
    .. automethod:: as_aff
    .. automethod:: as_basic_set
    .. automethod:: as_basic_map
    """
    active_dim_types: ClassVar[frozenset[DimType]] = frozenset(
        {DimType.param, DimType.in_, DimType.out})

    @staticmethod
    def equality_from_aff(aff: Aff) -> Constraint:
        return Constraint(
            isl.Constraint.equality_from_aff(aff._obj),
            aff.space.as_set_space().with_empty_dim_type(DimType.in_))

    @staticmethod
    def inequality_from_aff(aff: Aff) -> Constraint:
        return Constraint(
            isl.Constraint.inequality_from_aff(aff._obj),
            aff.space.as_set_space().with_empty_dim_type(DimType.in_))

    @property
    def is_equality(self):
        return self._obj.is_equality()

    def as_aff(self) -> Aff:
        if self.space.dim(DimType.in_):
            raise ValueError("cannot convert constraint with 'in' dimensions to aff")
        from .expression_like import Aff
        return Aff(
            self._obj.get_aff(),
            self.space
            .drop_dim_type(DimType.in_).move_dim_type(DimType.out, DimType.in_))

    def as_basic_set(self):
        return BasicSet(
            isl.BasicSet.universe(self._obj.get_space()) .add_constraint(self._obj),
            self.space.drop_dim_type(DimType.in_))

    def as_basic_map(self):
        return BasicMap(
            isl.BasicMap.universe(self._obj.get_space()) .add_constraint(self._obj),
            self.space)


def make_constraint(obj: isl.Constraint) -> Constraint:
    return Constraint(obj, Space.from_isl(obj, Set.active_dim_types))


class _NamedIslSetOrMapLike(NamedIslObject[IslSetOrMapLikeT_co]):
    __doc__ = """
    .. automethod:: is_empty
    .. automethod:: plain_is_empty
    .. automethod:: plain_is_universe
    .. automethod:: universe_like_me
    .. automethod:: empty_like_me
    .. automethod:: fix_dim
    .. automethod:: eliminate
    .. automethod:: project_out
    .. automethod:: project_out_except
    .. automethod:: gist
    .. automethod:: remove_divs
    .. automethod:: __and__
    .. automethod:: __or__
    .. automethod:: __sub__
    .. automethod:: __eq__
    .. automethod:: equals
    .. automethod:: __lt__
    .. automethod:: __le__
    """

    def is_empty(self) -> bool:
        return self._obj.is_empty()

    def plain_is_empty(self) -> bool:
        return self._obj.plain_is_empty()

    def plain_is_universe(self) -> bool:
        return self._obj.plain_is_universe()

    def universe_like_me(self) -> Self:
        return type(self)(type(self._obj).universe(self._obj.space), self.space)

    def empty_like_me(self) -> Self:
        return type(self)(type(self._obj).empty(self._obj.space), self.space)

    def fix_dim(self, name: str, value: isl.Val | int) -> Self:
        dt, idx = self.space.name_to_dim[name]
        return type(self)(
            cast("IslSetOrMapLikeT_co", self._obj.fix_val(dt.as_isl(), idx, value)),
            self.space,
        )

    def eliminate(self,
                names: Collection[str],
                *, cache: Cache | None = None
            ) -> Self:
        "Keeps the dimensions, but eliminates constraints."
        obj = self._obj
        for dt, chunks in chunked_dims_by_type(
                names, self.space.name_to_dim).items():
            for start, count in chunks:
                obj = with_cache(
                    cache, type(obj).eliminate, obj, dt.as_isl(), start, count)  # pyright: ignore[reportArgumentType]

        return type(self)(obj, self.space)  # pyright: ignore[reportArgumentType]

    def eliminate_except(
                self,
                names_to_keep: Collection[str],
                *, cache: Cache | None = None,
            ) -> Self:
        "Keeps the dimensions, but eliminates constraints."
        if isinstance(names_to_keep, str):
            raise TypeError("names_to_keep must be a collection of str")

        names_to_eliminate = [
            name for name in self.space.name_to_dim if name not in names_to_keep
        ]

        return self.eliminate(names_to_eliminate, cache=cache)

    def project_out(self,
                names: str | Collection[str],
                *, cache: Cache | None = None,
            ) -> Self:
        "Eliminates the dimensions and constraints."
        new_dimtype_to_names = {
            dt: list(names) for dt, names in self.space.dimtype_to_names.items()}
        obj = self._obj

        for dt, chunks in chunked_dims_by_type(
                names, self.space.name_to_dim).items():
            for start, count in chunks[::-1]:
                del new_dimtype_to_names[dt][start:start+count]
                obj = with_cache(
                    cache, type(obj).project_out, obj, dt.as_isl(), start, count)  # pyright: ignore[reportArgumentType]

        return type(self)(obj, Space(constantdict({  # pyright: ignore[reportArgumentType]
            dt: tuple(names)
            for dt, names in new_dimtype_to_names.items()
        })))

    def project_out_except(
                self,
                names_to_keep: Collection[str],
                *, cache: Cache | None = None,
            ) -> Self:
        "Eliminates the dimensions and constraints."
        if isinstance(names_to_keep, str):
            raise TypeError("names_to_keep must be a collection of str")

        names_to_project_out = [
            name for name in self.space.name_to_dim if name not in names_to_keep
        ]

        return self.project_out(names_to_project_out, cache=cache)

    def gist(self, context: Self) -> Self:
        self_aligned, context_aligned = align_two(self, context)
        return type(self)(
            self_aligned._obj.gist(context_aligned._obj),
            self_aligned.space,
        )

    def remove_divs(self) -> Self:
        return type(self)(
            cast("IslSetOrMapLikeT_co", self._obj.remove_divs()),
            self.space,
        )

    def __and__(self, other: Self) -> Self:
        return cast("Self", _align_and_apply_binary_op(self, other, operator.and_))

    def __or__(self, other: Self) -> Self:
        return cast("Self", _align_and_apply_binary_op(self, other, operator.or_))

    def __sub__(self, other: Self) -> Self:
        return cast("Self", _align_and_apply_binary_op(self, other, operator.sub))

    def equals(self, other: Self) -> bool:
        return _compare_set_or_map_like(self, other, operator.eq)

    def __lt__(self, other: Self) -> bool:
        return _compare_set_or_map_like(self, other, operator.lt)

    def __le__(self, other: Self) -> bool:
        return _compare_set_or_map_like(self, other, operator.le)


class _NamedIslSetLike(_NamedIslSetOrMapLike[IslSetLikeT]):
    __doc__ = """
    .. automethod:: is_bounded
    """

    active_dim_types: ClassVar[frozenset[DimType]] = frozenset(
        {DimType.param, DimType.out})

    if __debug__:
        def __post_init__(self) -> None:
            assert not self._obj.is_params()
            return super().__post_init__()

    def is_bounded(self) -> bool:
        return self._obj.is_bounded()


class _NamedIslUnbasic(_NamedIslSetOrMapLike[IslUnbasicT_co]):
    __doc__ = """
    .. automethod:: equate_dims
    .. automethod:: as_pw_multi_aff
    .. automethod:: remove_redundancies
    .. automethod:: coalesce
    """

    def equate_dims(
        self,
        names: Sequence[tuple[str, str]]
    ) -> Self:
        obj = self._obj
        for lhs_name, rhs_name in names:
            if lhs_name != rhs_name:
                lhs_dim_id = self.space.name_to_dim[lhs_name]
                rhs_dim_id = self.space.name_to_dim[rhs_name]
                obj = cast("IslUnbasicT_co", obj.equate(
                    lhs_dim_id.dim_type.as_isl(),
                    lhs_dim_id.dim_index,
                    rhs_dim_id.dim_type.as_isl(),
                    rhs_dim_id.dim_index,
                ))

        return type(self)(obj, self.space)

    def as_pw_multi_aff(self) -> PwMultiAff:
        return make_pw_multi_aff(self.as_isl().as_pw_multi_aff())

    def remove_redundancies(self):
        return type(self)(
            cast("IslUnbasicT_co", self._obj.remove_redundancies()), self.space)

    def coalesce(self) -> Self:
        return type(self)(cast("IslUnbasicT_co", self._obj.coalesce()), self.space)


class _NamedIslBasic(_NamedIslSetOrMapLike[IslBasicT_co]):
    __doc__ = """
    """


class BasicSet(_NamedIslSetLike[isl.BasicSet], _NamedIslBasic[isl.BasicSet]):
    __doc__ = f"""
    .. automethod:: add_constraint
    .. autoattribute:: var_affs
    .. automethod:: as_set
    .. automethod:: get_constraints
    {_NamedIslSetLike.__doc__}
    {_NamedIslBasic.__doc__}
    {_NamedIslSetOrMapLike.__doc__}
    """

    def add_constraint(self, cns: Constraint, /) -> BasicSet:
        if __debug__:
            if cns.space.dim(DimType.in_):
                raise ValueError("cannot add constraint with 'in' dimension to set")
            if not self.space.order_equals(cns.space.drop_dim_type(DimType.in_)):
                raise ValueError("spaces don't match")
        return BasicSet(self._obj.add_constraint(cns._obj), self.space)

    def get_constraints(self):
        return [
            Constraint(cns, self.space.with_empty_dim_type(DimType.in_))
            for cns in self._obj.get_constraints()]

    @cached_property
    def var_affs(self) -> Mapping[str | Literal[0], Aff]:
        r"""
        Returns a lazily-evaluated mapping from dimension names (or zero) to
        :class:`PwAff`\ s.

        .. note::

            Lazy evaluation means you do not pay for the creation of unused dimensions.
        """
        from .expression_like import affs_from_domain_space
        return affs_from_domain_space(self.space)

    def as_set(self) -> Set:
        return Set(self._obj.to_set(), self.space)


@overload
def make_basic_set(src: str, ctx: isl.Context | None = None) -> BasicSet: ...


@overload
def make_basic_set(src: isl.BasicSet) -> BasicSet: ...


def make_basic_set(src: str | isl.BasicSet, ctx: isl.Context | None = None) -> BasicSet:
    obj = isl.BasicSet(src, ctx) if isinstance(src, str) else src

    if obj.is_params():
        # shield the user from 'param domain' complexity
        obj = obj.from_params()

    return BasicSet(obj, Space.from_isl(obj, BasicSet.active_dim_types))


class Set(_NamedIslSetLike[isl.Set], _NamedIslUnbasic[isl.Set]):
    __doc__ = f"""
    .. automethod:: complement
    .. automethod:: simple_hull
    .. automethod:: convex_hull
    .. automethod:: get_basic_sets
    .. automethod:: dim_max
    .. automethod:: dim_min
    .. autoattribute:: var_pw_affs
    .. automethod:: as_map
    .. automethod:: as_basic
    {_NamedIslUnbasic.__doc__}
    {_NamedIslSetLike.__doc__}
    {_NamedIslSetOrMapLike.__doc__}
    """

    def complement(self) -> Set:
        return Set(self._obj.complement(), self.space)

    def simple_hull(self):
        return BasicSet(self._obj.simple_hull(), self.space)

    def convex_hull(self) -> BasicSet:
        return BasicSet(self._obj.convex_hull(), self.space)

    def get_basic_sets(self) -> list[BasicSet]:
        return [BasicSet(bs, self.space) for bs in self._obj.get_basic_sets()]

    def dim_max(self, name: str, *, cache: Cache | None = None) -> PwAff:
        dt, idx = self.space.name_to_dim[name]
        if dt != DimType.out:
            raise ValueError("can only take max with respect to set dimensions")
        return PwAff(with_cache(cache, isl.Set.dim_max, self._obj, idx),
            self.space.drop_dim_type(DimType.out).with_empty_dim_type(DimType.in_))

    def dim_min(self, name: str, *, cache: Cache | None = None) -> PwAff:
        dt, idx = self.space.name_to_dim[name]
        if dt != DimType.out:
            raise ValueError("can only take min with respect to set dimensions")
        return PwAff(with_cache(cache, isl.Set.dim_min, self._obj, idx),
            self.space.drop_dim_type(DimType.out).with_empty_dim_type(DimType.in_))

    @cached_property
    def var_pw_affs(self) -> Mapping[str | Literal[0], PwAff]:
        r"""
        Returns a lazily-evaluated mapping from dimension names (or zero)
        to :class:`PwAff`\ s.

        .. note::

            Lazy evaluation means you do not pay for the creation of unused dimensions.
        """
        from .expression_like import pw_affs_from_domain_space
        return pw_affs_from_domain_space(self.space)

    def as_map(self, in_names: Collection[str]) -> Map:
        result = isl.Map.universe(self._obj.space)
        result = result.intersect_range(self._obj)
        named_map = Map(result, Space(dimtype_to_names=constantdict({
            **self.space.dimtype_to_names,
            DimType.in_: ()
        })))
        return named_map.move_dims(in_names, DimType.in_)

    def as_basic(self) -> BasicSet:
        """Will only succeed if self consists of a single piece.
        Try :meth:`coalesce` if not.
        """
        bss = self._obj.get_basic_sets()
        if len(bss) > 1:
            raise ValueError("set has multiple basic sets")
        return BasicSet(bss[0], self.space)


@overload
def make_set(src: str, ctx: isl.Context | None = None) -> Set: ...


@overload
def make_set(src: isl.Set) -> Set: ...


def make_set(src: isl.Set | str, ctx: isl.Context | None = None) -> Set:
    obj = isl.Set(src, ctx) if isinstance(src, str) else src

    if obj.is_params():
        # shield the user from 'param domain' complexity
        obj = obj.from_params()

    return Set(obj, Space.from_isl(obj, Set.active_dim_types))


class _NamedIslMapLike(_NamedIslSetOrMapLike[IslMapLikeT]):
    __doc__ = """
    .. autoattribute:: active_dim_types
    .. automethod:: reverse
    """
    active_dim_types: ClassVar[frozenset[DimType]] = frozenset(
        {DimType.param, DimType.in_, DimType.out})

    def reverse(self) -> Self:
        return type(self)(
            cast("IslMapLikeT", self._obj.reverse()),
            self.space.swap_dim_types(DimType.in_, DimType.out))


class BasicMap(_NamedIslMapLike[isl.BasicMap], _NamedIslBasic[isl.BasicMap]):
    __doc__ = f"""
    .. automethod:: domain
    .. automethod:: range
    .. automethod:: intersect_domain
    .. automethod:: intersect_range
    .. automethod:: get_constraints
    {_NamedIslMapLike.__doc__}
    {_NamedIslBasic.__doc__}
    {_NamedIslSetOrMapLike.__doc__}
    """

    def add_constraint(self, cns: Constraint, /) -> BasicMap:
        if __debug__:  # ruff:ignore[collapsible-if]
            if not self.space.order_equals(cns.space):
                raise ValueError("spaces don't match")
        return BasicMap(self._obj.add_constraint(cns._obj), self.space)

    def get_constraints(self):
        return [
            Constraint(cns, self.space) for cns in self._obj.get_constraints()]

    def domain(self) -> BasicSet:
        return BasicSet(
            self._obj.domain(),
            self.space
                .drop_dim_type(DimType.out)
                .move_dim_type(DimType.in_, DimType.out)
        )

    def range(self) -> BasicSet:
        return BasicSet(self._obj.range(), self.space.drop_dim_type(DimType.in_))

    def intersect_domain(self, domain: BasicSet) -> Self:
        self_a, domain_a = align_for_compostition(
            self, DimType.in_, domain, DimType.out)
        return type(self)(self_a._obj.intersect_domain(domain_a._obj), self_a.space)

    def intersect_range(self, range: BasicSet) -> Self:
        self_a, range_a = align_for_compostition(
            self, DimType.out, range, DimType.out)
        return type(self)(self_a._obj.intersect_range(range_a._obj), self_a.space)


@overload
def make_basic_map(src: str, ctx: isl.Context | None = None) -> BasicMap: ...


@overload
def make_basic_map(src: isl.BasicMap) -> BasicMap: ...


def make_basic_map(src: str | isl.BasicMap, ctx: isl.Context | None = None) -> BasicMap:
    obj = isl.BasicMap(src, ctx) if isinstance(src, str) else src
    return BasicMap(obj, Space.from_isl(obj, BasicMap.active_dim_types))


def make_map_from_domain_and_range(
    domain: BasicSet | Set, range_: BasicSet | Set
) -> BasicMap | Map:
    if isinstance(domain, BasicSet) and isinstance(range_, BasicSet):
        domain_obj = domain.as_isl()
        range_obj = range_.as_isl()
        assert isinstance(domain_obj, isl.BasicSet)
        assert isinstance(range_obj, isl.BasicSet)
        return make_basic_map(
            isl.BasicMap.from_domain_and_range(
                domain_obj,
                range_obj,
            )
        )

    domain_obj = domain.as_isl()
    range_obj = range_.as_isl()
    assert isinstance(domain_obj, isl.BasicSet | isl.Set)
    assert isinstance(range_obj, isl.BasicSet | isl.Set)
    return make_map(
        isl.Map.from_domain_and_range(
            domain_obj,
            range_obj,
        )
    )


class Map(_NamedIslMapLike[isl.Map], _NamedIslUnbasic[isl.Map]):
    __doc__ = f"""
    .. automethod:: complement
    .. automethod:: simple_hull
    .. automethod:: convex_hull
    .. automethod:: get_basic_maps
    .. automethod:: domain
    .. automethod:: range
    .. automethod:: intersect_domain
    .. automethod:: intersect_range
    .. automethod:: apply_range
    .. automethod:: apply_domain
    .. automethod:: as_basic
    .. automethod:: as_set
    .. autoattribute:: domain_var_pw_affs
    .. autoattribute:: range_var_pw_affs
    {_NamedIslSetOrMapLike.__doc__}
    {_NamedIslUnbasic.__doc__}
    {_NamedIslMapLike.__doc__}
    """

    def complement(self) -> Map:
        return Map(self._obj.complement(), self.space)

    def simple_hull(self):
        return BasicMap(self._obj.simple_hull(), self.space)

    def convex_hull(self) -> BasicMap:
        return BasicMap(self._obj.convex_hull(), self.space)

    def get_basic_maps(self) -> list[BasicMap]:
        return [BasicMap(bs, self.space) for bs in self._obj.get_basic_maps()]

    def domain(self) -> Set:
        return Set(
            self._obj.domain(),
            self.space
                .drop_dim_type(DimType.out)
                .move_dim_type(DimType.in_, DimType.out)
        )

    def range(self) -> Set:
        return Set(self._obj.range(), self.space.drop_dim_type(DimType.in_))

    def intersect_domain(self, domain: Set) -> Self:
        self_a, domain_a = align_for_compostition(
            self, DimType.in_, domain, DimType.out)
        return type(self)(self_a._obj.intersect_domain(domain_a._obj), self_a.space)

    def intersect_range(self, range: Set) -> Self:
        self_a, range_a = align_for_compostition(
            self, DimType.out, range, DimType.out)
        return type(self)(self_a._obj.intersect_range(range_a._obj), self_a.space)

    def apply_range(self, other: Self) -> Self:
        self_a, other_a = align_for_compostition(self, DimType.out, other, DimType.in_)
        return type(self)(
            self_a._obj.apply_range(other_a._obj),
            Space(constantdict({
                DimType.param: self_a.space.dimtype_to_names[DimType.param],
                DimType.in_: self_a.space.dimtype_to_names[DimType.in_],
                DimType.out: other_a.space.dimtype_to_names[DimType.out],
            })))

    def apply_domain(self, other: Self) -> Self:
        self_a, other_a = align_for_compostition(self, DimType.in_, other, DimType.out)
        return type(self)(
            self_a._obj.apply_domain(other_a._obj),
            Space(constantdict({
                DimType.param: self_a.space.dimtype_to_names[DimType.param],
                DimType.in_: other_a.space.dimtype_to_names[DimType.in_],
                DimType.out: self.space.dimtype_to_names[DimType.out],
            })))

    def as_basic(self) -> BasicMap:
        """Will only succeed if self consists of a single piece.
        Try :meth:`coalesce` if not.
        """
        bms = self._obj.get_basic_maps()
        if len(bms) > 1:
            raise ValueError("set has multiple basic sets")
        return BasicMap(bms[0], self.space)

    def as_set(self) -> Set:
        result = self.move_dims(self.space.in_names, DimType.out)
        return result.range()

    @cached_property
    def domain_var_pw_affs(self) -> Mapping[str | Literal[0], PwAff]:
        r"""
        Returns a lazily-evaluated mapping from dimension names (or zero)
        to :class:`PwAff`\ s for the domain variables of *self*

        .. note::

            Lazy evaluation means you do not pay for the creation of unused dimensions.
        """
        from .expression_like import pw_affs_from_domain_space
        return pw_affs_from_domain_space(
            self.space
            .drop_dim_type(DimType.out)
            .move_dim_type(DimType.in_, DimType.out))

    @cached_property
    def range_var_pw_affs(self) -> Mapping[str | Literal[0], PwAff]:
        r"""
        Returns a lazily-evaluated mapping from dimension names (or zero)
        to :class:`PwAff`\ s for the domain variables of *self*

        .. note::

            Lazy evaluation means you do not pay for the creation of unused dimensions.
        """
        from .expression_like import pw_affs_from_domain_space
        return pw_affs_from_domain_space(self.space.drop_dim_type(DimType.out))


@overload
def make_map(src: str, ctx: isl.Context | None = None) -> Map: ...


@overload
def make_map(src: isl.Map) -> Map: ...


def make_map(src: str | isl.Map, ctx: isl.Context | None = None) -> Map:
    obj = isl.Map(src, ctx) if isinstance(src, str) else src
    return Map(obj, Space.from_isl(obj, Map.active_dim_types))
