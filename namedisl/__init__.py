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
    Map,
    Set,
    make_basic_map,
    make_basic_set,
    make_map,
    make_set,
)


__all__ = [
    "Aff",
    "BasicMap",
    "BasicSet",
    "Map",
    "MultiAff",
    "PwAff",
    "PwMultiAff",
    "PwQPolynomial",
    "QPolynomial",
    "Set",
    "make_aff",
    "make_basic_map",
    "make_basic_set",
    "make_map",
    "make_multi_aff",
    "make_pw_aff",
    "make_pw_multi_aff",
    "make_pw_qpolynomial",
    "make_qpolynomial",
    "make_set"
]
