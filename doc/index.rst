Welcome to namedisl's documentation!
====================================

:mod:`namedisl` is a Python library for (name-based) polyhedral operations,
based on Sven Verdoolaege's `isl <https://libisl.sourceforge.io/>`__.

Unlike in isl, names matter in namedisl. Observe:

.. doctest::

    >>> import namedisl as nisl
    >>> print(nisl.make_set("{[i]: 0<=i<10}") & nisl.make_set("{[j]: 0<=j<10}"))
    { [i, j] : 0 <= i <= 9 and 0 <= j <= 9 }

I.e., since `i` and `j` have different names, they are considered different dimensions.
This contrasts against the behavior of :mod:`islpy` (and upstream isl),
which reason by dimension index and would consider ``i`` and ``j`` the
same axis (in most cases).


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    overview
    ref_set
    ref_expr
    ref_core
    misc

    🚀 Github <https://github.com/inducer/namedisl>
    💾 Download Releases <https://pypi.org/project/namedisl>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
