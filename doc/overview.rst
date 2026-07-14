About ``namedisl``
==================
.. module:: namedisl

Polyhedral model
----------------

The polyhedral model permits precise reasoning about subsets of :math:`\mathbb{Z}^n`.
A subset is expressed as an intersection of affine constraints:

.. math::

    \begin{align*}
    a_1 z_1+ \cdots + a_n z_n + c &\ge 0, \quad \text{or}\\
    a_1 z_1+ \cdots + a_n z_n + c &= 0,
    \end{align*}

with :math:`a_i,c\in\mathbb Z` fixed coefficients and :math:`z_i`
the dimensions of the set.

An intersection of these hyperplanes or half-spaces is a
:class:`BasicSet`. Individual dimensions may be
marked as 'existentially quantified' (or equivalently, 'projected out', cf.
:meth:`BasicSet.project_out`).  In the absence of existentially quantified
variables, the resulting set is *convex*, in their presence it is called
*quasi-convex*.

Other dimensions may be marked as 'parameters' (cf. :class:`DimType.param`). 
Observe that this has the effect of allowing size-parametric set operations,
allowing operation on an infinite, symbolically-defined family of sets.

A union of :class:`BasicSet`\ s is a :class:`Set`, which is not necessarily
convex.

:class:`Map`\  s (with their analogous :class:`BasicMap`\ s) are mathematically
the same as sets, but permit labeling some of their dimensions as "inputs"
(cf. :class:`DimType.in_`), making them suited to represent relations
(such ass data access or dependency).

An :class:`Aff` is a quasi-affine expression as above, with existentially
quantified variables expressed in the form of modular arithmetic.

A :class:`PwAff` is a piecewise quasi-affine expression, consisting of tuples ``(set, aff)``,
with each :class:`Set` indicating where the corresponding :class:`Aff` defines the value
of the expression.

``namedisl`` interface guide
----------------------------

Immutability
^^^^^^^^^^^^

Every object in :mod:`namedisl` should be considered immutable. Even if
mutation is technically possible via Python gymnastics, the results
of doing so are undefined.

All objects are hashable, with equality semantics defined as below.

Equality
^^^^^^^^
The ``==`` operator uses the cheapest and strictest available comparison.
It is mainly intended for use in hashing/ cache retrieval.

Mathematical equivalence may be available as a separate, likely more expensive
operation (cf. :meth:`Set.equals`).

Dimension names are required to match in order for equality to hold.
To keep comparison inexpensive, dimension order in the underlying
isl object (while not otherwise exposed) is also required to
match for equality.

Names matter
^^^^^^^^^^^^

Consider the sets ``{[i]: 0<=i<10}`` and ``{[j]: 0<=j<10}`` the same set.
Observe

.. doctest::

    >>> import namedisl as nisl
    >>> print(nisl.make_set("{[i]: 0<=i<10}") & nisl.make_set("{[j]: 0<=j<10}"))
    { [i, j] : 0 <= i <= 9 and 0 <= j <= 9 }

I.e., since `i` and `j` have different names, they are considered different dimensions.

:mod:`namedisl` keeps name information in :class:`Space` objects that are shared
when possible.

Dimension order is an implementation detail
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With the exception of hashing and ``==`` equality, dimension order
(even if it can be observed by examining the :class:`Space`
or through conversion to :class:`str`) is an implementation detail that
does not influence the result of an operation. Dimension order
of internal ISL objects is not guaranteed to be stable 
and may change without notice.

Constructors are private
^^^^^^^^^^^^^^^^^^^^^^^^

Object instances should be created via ``make_*`` functions. Constructors
should never be directly called by user code.
