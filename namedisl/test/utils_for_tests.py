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

import islpy as isl
import namedisl as nisl
from random import randint

from typing import Tuple

NamedSetReturnT = Tuple[nisl.Set, str, str]
NamedMapReturnT = Tuple[nisl.Map, NamedSetReturnT, NamedSetReturnT]

def generate_random_named_set(
        ndims: int, 
        dim_prefix: str,
        param: str | None
        ) -> NamedSetReturnT:
    dims = [f"{dim_prefix}_{i}" for i in range(ndims)]
    dim_str = ",".join(d for d in dims)

    if param is not None:
        conditions = f"0 <= {dim_str} < {param}"
        set_str = f"[{param}] -> {{ [{dim_str}] : {conditions} }}"
    else:
        upper_bounds = [randint(1, 100) for _ in range(ndims)]
        lower_bounds = [
            randint(0, upper_bound - 1) for upper_bound in upper_bounds]

        conditions = " and ".join(
                f"{lower_bound} <= {d} < {upper_bound}"
                for d, lower_bound, upper_bound in zip(dims, lower_bounds,
                                                       upper_bounds))
        set_str = f"{{ [{dim_str}] : {conditions} }}"

    return nisl.make_set(set_str), dim_str, conditions 


def generate_random_named_map(
        ndims_domain: int, 
        domain_prefix: str,
        domain_param: str | None,
        ndims_range: int,
        range_prefix: str,
        range_param: str | None
        ) -> NamedMapReturnT:

    d = generate_random_named_set(ndims_domain, domain_prefix, domain_param)
    r = generate_random_named_set(ndims_range, range_prefix, range_param)

    return (
        nisl.make_map(isl.Map.from_domain_and_range(d[0]._obj, r[0]._obj)),
        d,
        r
    )


