Reference
=========

.. automodule:: namedisl

Set and Map Objects
-------------------

.. autofunction:: make_basic_set

.. autofunction:: make_set

.. autofunction:: make_basic_map

.. autofunction:: make_map

.. autofunction:: make_map_from_domain_and_range

.. autoclass:: BasicSet
    :members:
    :inherited-members:
    :special-members: __and__, __or__, __sub__, __lt__, __le__, __gt__, __ge__
    :exclude-members: __init__, __new__

.. autoclass:: Set
    :members:
    :inherited-members:
    :special-members: __and__, __or__, __sub__, __lt__, __le__, __gt__, __ge__
    :exclude-members: __init__, __new__

.. autoclass:: BasicMap
    :members:
    :inherited-members:
    :special-members: __and__, __or__, __sub__, __lt__, __le__, __gt__, __ge__
    :exclude-members: __init__, __new__

.. autoclass:: Map
    :members:
    :inherited-members:
    :special-members: __and__, __or__, __sub__, __lt__, __le__, __gt__, __ge__
    :exclude-members: __init__, __new__

Expression Objects
------------------

.. autofunction:: make_aff

.. autofunction:: make_pw_aff

.. autofunction:: make_qpolynomial

.. autofunction:: make_pw_qpolynomial

.. autofunction:: make_multi_aff

.. autofunction:: make_pw_multi_aff

.. autoclass:: Aff
    :members:
    :inherited-members:
    :special-members: __add__, __sub__, __mul__
    :exclude-members: __init__, __new__

.. autoclass:: PwAff
    :members:
    :inherited-members:
    :special-members: __add__, __sub__, __mul__
    :exclude-members: __init__, __new__

.. autoclass:: QPolynomial
    :members:
    :inherited-members:
    :special-members: __add__, __sub__, __mul__
    :exclude-members: __init__, __new__

.. autoclass:: PwQPolynomial
    :members:
    :inherited-members:
    :special-members: __add__, __sub__, __mul__
    :exclude-members: __init__, __new__

.. autoclass:: MultiAff
    :members:
    :inherited-members:
    :exclude-members: __init__, __new__

.. autoclass:: PwMultiAff
    :members:
    :inherited-members:
    :exclude-members: __init__, __new__
