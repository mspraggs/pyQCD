from cpython cimport Py_buffer
from libcpp.vector cimport vector

import numpy as np

cimport atomics
cimport core
from core cimport {% for td in typedefs %}_{{ td.name }}, {{ td.name }}{% if not loop.last %}, {% endif %}{% endfor %}


{% for typedef in typedefs %}
    {% with typedef=typedef %}

        {% include typedef.impl_template %}

    {% endwith %}
{% endfor %}