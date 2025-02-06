namedisl: Integer Set Library with Significant Names
====================================================

.. image:: https://gitlab.tiker.net/inducer/namedisl/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/namedisl/commits/main
.. image:: https://github.com/inducer/namedisl/actions/workflows/ci.yml/badge.svg
    :alt: Github Build Status
    :target: https://github.com/inducer/namedisl/actions/workflows/ci.yml
.. image:: https://badge.fury.io/py/namedisl.svg
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/namedisl/

namedisl is a Python wrapper around Sven Verdoolaege's `isl
<https://libisl.sourceforge.io/>`_, a library for manipulating sets and
relations of integer points bounded by linear constraints.

It is based on `islpy <https://github.com/inducer/islpy>`__ (and, transitively, on
`isl <https://libisl.sourceforge.io/>`__), with the important distinction that *names
matter*. While isl (and hence islpy) has support for naming set dimensions, the names do
not matter: isl considers ``{[i]: 0<=i<10}`` and ``{[j]: 0<=j<10}`` the same set.
Relatedly, isl's  (and hence islpy's) interface is based around dimension indices
rather than names. Meanwhile, namedisl's interface uses names to identify axes.

Supported operations on sets include

* intersection, union, set difference,
* emptiness check,
* convex hull,
* (integer) affine hull,
* integer projection,
* computing the lexicographic minimum using parametric integer programming,
* coalescing, and
* parametric vertex enumeration.

It also includes an ILP solver based on generalized basis reduction, transitive
closures on maps (which may encode infinite graphs), dependence analysis and
bounds on piecewise step-polynomials.

namedisl comes with comprehensive `documentation <http://documen.tician.de/namedisl>`_.

