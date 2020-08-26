:orphan:

===
API
===

.. jinja::

   {% set functions = ['plot_microstructures', 'generate_delta', 'generate_multiphase', 'generate_checkerboard', 'solve_cahn_hilliard', 'solve_fe', 'coeff_to_real'] | sort %}

   {% set classes = ['PrimitiveTransformer', 'LegendreTransformer', 'TwoPointCorrelation', 'FlattenTransformer', 'LocalizationRegressor', 'ReshapeTransformer'] | sort %}

   .. currentmodule:: pymks

   .. autosummary::
   {% for function in functions %}
       {{ function }}
   {% endfor %}
   {% for class in classes %}
       {{ class }}
   {% endfor %}

   {% for function in functions %}
   .. autofunction:: pymks.{{ function }}

   {% endfor %}


   {% for class in classes %}
   .. autoclass:: pymks.{{ class }}
       :members:

   {% endfor %}
