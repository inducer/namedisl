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

import pytest

from .utils_for_tests import generate_random_named_set, get_name_sequence


@pytest.mark.parametrize("ndims", [2, 3, 4, 5])
@pytest.mark.parametrize("has_params", [True, False])
def test_names(ndims: int, has_params: bool):
    s_param = "n" if has_params else None
    s, s_dims, _ = generate_random_named_set(ndims, "s", s_param)
    names = frozenset(s_dims.split(","))

    if s_param:
        names = names | frozenset({s_param})

    assert s.names == names


@pytest.mark.parametrize("ndims", [2, 3, 4, 5])
@pytest.mark.parametrize("n_names_to_add", [2, 3, 4, 5])
def test_add_names(
        ndims: int,
        n_names_to_add: int
    ):

    s, _s_dims, _ = generate_random_named_set(ndims, "s", None)
    new_set_names, _ = get_name_sequence(n_names_to_add, "set")

    from namedisl.tags import SetName
    s = s.add_names([SetName(name) for name in new_set_names])

    print(s)
