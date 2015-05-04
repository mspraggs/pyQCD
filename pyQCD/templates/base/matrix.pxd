from libcpp cimport complex

cdef extern from "Eigen/Dense" namespace "Eigen":
    cdef cppclass {{ matrixdef.matrix_name }} "Matrix<std::complex<{{ precision }}>, {{ matrixdef.num_rows }}, {{ matrixdef.num_cols }}>":
        {{ matrixdef.matrix_name }}() except +
        {{ matrixdef.matrix_name }} adjoint()
        {% if matrixdef.num_cols > 1 %}
        const complex[{{ precision }}]& operator()(int, int) except +
        {% else %}
        const complex[{{ precision }}]& operator[](int) except +
        {% endif %}