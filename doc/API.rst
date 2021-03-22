:orphan:

===
API
===

.. toctree::
   :maxdepth: 2


.. jinja::

   Table
   =====

   {% set functions = ['plot_microstructures', 'generate_delta', 'generate_multiphase', 'generate_checkerboard', 'solve_cahn_hilliard', 'solve_fe', 'coeff_to_real', 'paircorr_from_twopoint', 'two_point_stats', 'correlations_multiple', 'test] | sort %}


   {% set classes = ['PrimitiveTransformer', 'LegendreTransformer', 'TwoPointCorrelation', 'FlattenTransformer', 'LocalizationRegressor', 'ReshapeTransformer', 'GenericTransformer'] | sort %}

   .. currentmodule:: pymks

   .. autosummary::
   {% for function in functions %}
       {{ function }}
   {% endfor %}
   {% for class in classes %}
       {{ class }}
   {% endfor %}

   Functions
   =========

   {% for function in functions %}
   .. autofunction:: pymks.{{ function }}

   {% endfor %}

   Classes
   =======

   {% for class in classes %}
   .. autoclass:: pymks.{{ class }}
       :members:

   {% endfor %}
