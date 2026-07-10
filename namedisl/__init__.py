"""
Name-aware wrappers for :mod:`islpy` objects.

namedisl offers small Python wrappers around isl sets, maps, and expression
objects.  The wrappers keep a separate mapping from dimension names to isl
dimension positions, align operands by name before applying binary operations,
and reconstruct ordinary :mod:`islpy` objects when callers need to interoperate
with islpy or downstream libraries.

Most users should construct objects through the ``make_*`` functions exported
from this module.
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

from typing import overload

from .core import DimType, IslObject, Space
from .expression_like import (
    Aff,
    MultiAff,
    PwAff,
    PwMultiAff,
    PwQPolynomial,
    QPolynomial,
    make_aff,
    make_multi_aff,
    make_pw_aff,
    make_pw_multi_aff,
    make_pw_qpolynomial,
    make_qpolynomial,
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
]


import islpy as isl


_ISL_TYPE_TO_CONSTRUCTOR = {
    isl.Aff: make_aff,
    isl.QPolynomial: make_qpolynomial,
    isl.PwAff: make_pw_aff,
    isl.PwQPolynomial: make_pw_qpolynomial,
    isl.MultiAff: make_multi_aff,
    isl.PwMultiAff: make_pw_multi_aff,
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
