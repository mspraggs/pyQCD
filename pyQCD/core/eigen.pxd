cdef extern from "Eigen/Dense" namespace "Eigen":
    cdef cppclass ColourMatrix "Matrix<std::complex<double>, 3, 3>":
        ColourMatrix();

    cdef ColourMatrix colour_matrix_zeros "Matrix<std::complex<double>, 3, 3>::Zeros"()
    cdef ColourMatrix colour_matrix_ones "Matrix<std::complex<double>, 3, 3>::Ones"()
    cdef ColourMatrix colour_matrix_identity "Matrix<std::complex<double>, 3, 3>::Identity"()