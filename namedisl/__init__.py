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

from typing import overload

from .core import Cache, DimType, IslObject, Space, align_two
from .expression_like import (
    Aff,
    MultiAff,
    PwAff,
    PwMultiAff,
    PwQPolynomial,
    QPolynomial,
    affs_from_domain_space,
    make_aff,
    make_multi_aff,
    make_pw_aff,
    make_pw_multi_aff,
    make_pw_qpolynomial,
    make_qpolynomial,
    pw_affs_from_domain_space,
)
from .set_like import (
    BasicMap,
    BasicSet,
    Constraint,
    Map,
    Set,
    make_basic_map,
    make_basic_set,
    make_constraint,
    make_map,
    make_map_from_domain_and_range,
    make_set,
)


__all__ = [
    "Aff",
    "BasicMap",
    "BasicSet",
    "Cache",
    "Constraint",
    "DimType",
    "Map",
    "MultiAff",
    "PwAff",
    "PwMultiAff",
    "PwQPolynomial",
    "QPolynomial",
    "Set",
    "Space",
    "affs_from_domain_space",
    "align_two",
    "make_aff",
    "make_basic_map",
    "make_basic_set",
    "make_constraint",
    "make_map",
    "make_map_from_domain_and_range",
    "make_multi_aff",
    "make_pw_aff",
    "make_pw_multi_aff",
    "make_pw_qpolynomial",
    "make_qpolynomial",
    "make_set",
    "pw_affs_from_domain_space",
]


import islpy as isl


_ISL_TYPE_TO_CONSTRUCTOR = {
    isl.Aff: make_aff,
    isl.QPolynomial: make_qpolynomial,
    isl.PwAff: make_pw_aff,
    isl.PwQPolynomial: make_pw_qpolynomial,
    isl.MultiAff: make_multi_aff,
    isl.PwMultiAff: make_pw_multi_aff,
    isl.Constraint: make_constraint,
    isl.BasicSet: make_basic_set,
    isl.Set: make_set,
    isl.BasicMap: make_basic_map,
    isl.Map: make_map,
}


@overload
def to_named(obj: isl.Aff) -> Aff: ...
@overload
def to_named(obj: isl.QPolynomial) -> QPolynomial: ...
@overload
def to_named(obj: isl.PwAff) -> PwAff: ...
@overload
def to_named(obj: isl.PwQPolynomial) -> PwQPolynomial: ...
@overload
def to_named(obj: isl.MultiAff) -> MultiAff: ...
@overload
def to_named(obj: isl.PwMultiAff) -> PwMultiAff: ...
@overload
def to_named(obj: isl.BasicSet) -> BasicSet: ...
@overload
def to_named(obj: isl.Set) -> Set: ...
@overload
def to_named(obj: isl.BasicMap) -> BasicMap: ...
@overload
def to_named(obj: isl.Map) -> Map: ...
@overload
def to_named(obj: isl.Constraint) -> Constraint: ...


def to_named(obj: IslObject) -> (
    Aff | QPolynomial
    | PwAff | PwQPolynomial
    | MultiAff | PwMultiAff
    | BasicSet | Set
    | BasicMap | Map
    | Constraint
):
    return _ISL_TYPE_TO_CONSTRUCTOR[type(obj)](obj)  # pyright: ignore[reportCallIssue, reportUnknownVariableType, reportArgumentType]
