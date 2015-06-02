cimport complex
cimport colour_matrix
cimport colour_matrix_array
cimport lattice_colour_matrix
cimport gauge_field
cimport colour_vector
cimport fermion
cimport lattice_colour_vector
cimport fermion_field


cdef extern from "types.hpp":
    colour_matrix.ColourMatrix operator*(const double&, const colour_matrix.ColourMatrix&) except +
    colour_matrix.ColourMatrix operator*(const colour_matrix.ColourMatrix&, const double&) except +
    colour_matrix.ColourMatrix operator/(const colour_matrix.ColourMatrix&, const double&) except +
    colour_matrix.ColourMatrix operator*(const complex.Complex&, const colour_matrix.ColourMatrix&) except +
    colour_matrix.ColourMatrix operator*(const colour_matrix.ColourMatrix&, const complex.Complex&) except +
    colour_matrix.ColourMatrix operator/(const colour_matrix.ColourMatrix&, const complex.Complex&) except +
    colour_matrix_array.ColourMatrixArray operator*(const double&, const colour_matrix_array.ColourMatrixArray&) except +
    colour_matrix_array.ColourMatrixArray operator*(const colour_matrix_array.ColourMatrixArray&, const double&) except +
    colour_matrix_array.ColourMatrixArray operator/(const colour_matrix_array.ColourMatrixArray&, const double&) except +
    colour_matrix_array.ColourMatrixArray operator*(const complex.Complex&, const colour_matrix_array.ColourMatrixArray&) except +
    colour_matrix_array.ColourMatrixArray operator*(const colour_matrix_array.ColourMatrixArray&, const complex.Complex&) except +
    colour_matrix_array.ColourMatrixArray operator/(const colour_matrix_array.ColourMatrixArray&, const complex.Complex&) except +
    lattice_colour_matrix.LatticeColourMatrix operator*(const double&, const lattice_colour_matrix.LatticeColourMatrix&) except +
    lattice_colour_matrix.LatticeColourMatrix operator*(const lattice_colour_matrix.LatticeColourMatrix&, const double&) except +
    lattice_colour_matrix.LatticeColourMatrix operator/(const lattice_colour_matrix.LatticeColourMatrix&, const double&) except +
    lattice_colour_matrix.LatticeColourMatrix operator*(const complex.Complex&, const lattice_colour_matrix.LatticeColourMatrix&) except +
    lattice_colour_matrix.LatticeColourMatrix operator*(const lattice_colour_matrix.LatticeColourMatrix&, const complex.Complex&) except +
    lattice_colour_matrix.LatticeColourMatrix operator/(const lattice_colour_matrix.LatticeColourMatrix&, const complex.Complex&) except +
    gauge_field.GaugeField operator*(const double&, const gauge_field.GaugeField&) except +
    gauge_field.GaugeField operator*(const gauge_field.GaugeField&, const double&) except +
    gauge_field.GaugeField operator/(const gauge_field.GaugeField&, const double&) except +
    gauge_field.GaugeField operator*(const complex.Complex&, const gauge_field.GaugeField&) except +
    gauge_field.GaugeField operator*(const gauge_field.GaugeField&, const complex.Complex&) except +
    gauge_field.GaugeField operator/(const gauge_field.GaugeField&, const complex.Complex&) except +
    colour_vector.ColourVector operator*(const double&, const colour_vector.ColourVector&) except +
    colour_vector.ColourVector operator*(const colour_vector.ColourVector&, const double&) except +
    colour_vector.ColourVector operator/(const colour_vector.ColourVector&, const double&) except +
    colour_vector.ColourVector operator*(const complex.Complex&, const colour_vector.ColourVector&) except +
    colour_vector.ColourVector operator*(const colour_vector.ColourVector&, const complex.Complex&) except +
    colour_vector.ColourVector operator/(const colour_vector.ColourVector&, const complex.Complex&) except +
    fermion.Fermion operator*(const double&, const fermion.Fermion&) except +
    fermion.Fermion operator*(const fermion.Fermion&, const double&) except +
    fermion.Fermion operator/(const fermion.Fermion&, const double&) except +
    fermion.Fermion operator*(const complex.Complex&, const fermion.Fermion&) except +
    fermion.Fermion operator*(const fermion.Fermion&, const complex.Complex&) except +
    fermion.Fermion operator/(const fermion.Fermion&, const complex.Complex&) except +
    lattice_colour_vector.LatticeColourVector operator*(const double&, const lattice_colour_vector.LatticeColourVector&) except +
    lattice_colour_vector.LatticeColourVector operator*(const lattice_colour_vector.LatticeColourVector&, const double&) except +
    lattice_colour_vector.LatticeColourVector operator/(const lattice_colour_vector.LatticeColourVector&, const double&) except +
    lattice_colour_vector.LatticeColourVector operator*(const complex.Complex&, const lattice_colour_vector.LatticeColourVector&) except +
    lattice_colour_vector.LatticeColourVector operator*(const lattice_colour_vector.LatticeColourVector&, const complex.Complex&) except +
    lattice_colour_vector.LatticeColourVector operator/(const lattice_colour_vector.LatticeColourVector&, const complex.Complex&) except +
    fermion_field.FermionField operator*(const double&, const fermion_field.FermionField&) except +
    fermion_field.FermionField operator*(const fermion_field.FermionField&, const double&) except +
    fermion_field.FermionField operator/(const fermion_field.FermionField&, const double&) except +
    fermion_field.FermionField operator*(const complex.Complex&, const fermion_field.FermionField&) except +
    fermion_field.FermionField operator*(const fermion_field.FermionField&, const complex.Complex&) except +
    fermion_field.FermionField operator/(const fermion_field.FermionField&, const complex.Complex&) except +

    colour_matrix.ColourMatrix operator*(const colour_matrix.ColourMatrix&, const colour_matrix.ColourMatrix&) except +
    colour_matrix.ColourMatrix operator+(const colour_matrix.ColourMatrix&, const colour_matrix.ColourMatrix&) except +
    colour_matrix.ColourMatrix operator-(const colour_matrix.ColourMatrix&, const colour_matrix.ColourMatrix&) except +
    colour_matrix_array.ColourMatrixArray operator*(const colour_matrix.ColourMatrix&, const colour_matrix_array.ColourMatrixArray&) except +
    colour_matrix_array.ColourMatrixArray operator+(const colour_matrix.ColourMatrix&, const colour_matrix_array.ColourMatrixArray&) except +
    lattice_colour_matrix.LatticeColourMatrix operator*(const colour_matrix.ColourMatrix&, const lattice_colour_matrix.LatticeColourMatrix&) except +
    lattice_colour_matrix.LatticeColourMatrix operator+(const colour_matrix.ColourMatrix&, const lattice_colour_matrix.LatticeColourMatrix&) except +
    gauge_field.GaugeField operator*(const colour_matrix.ColourMatrix&, const gauge_field.GaugeField&) except +
    gauge_field.GaugeField operator+(const colour_matrix.ColourMatrix&, const gauge_field.GaugeField&) except +
    colour_matrix_array.ColourMatrixArray operator*(const colour_matrix_array.ColourMatrixArray&, const colour_matrix.ColourMatrix&) except +
    colour_matrix_array.ColourMatrixArray operator+(const colour_matrix_array.ColourMatrixArray&, const colour_matrix.ColourMatrix&) except +
    colour_matrix_array.ColourMatrixArray operator-(const colour_matrix_array.ColourMatrixArray&, const colour_matrix.ColourMatrix&) except +
    colour_matrix_array.ColourMatrixArray operator*(const colour_matrix_array.ColourMatrixArray&, const colour_matrix_array.ColourMatrixArray&) except +
    colour_matrix_array.ColourMatrixArray operator+(const colour_matrix_array.ColourMatrixArray&, const colour_matrix_array.ColourMatrixArray&) except +
    colour_matrix_array.ColourMatrixArray operator-(const colour_matrix_array.ColourMatrixArray&, const colour_matrix_array.ColourMatrixArray&) except +
    lattice_colour_matrix.LatticeColourMatrix operator*(const lattice_colour_matrix.LatticeColourMatrix&, const colour_matrix.ColourMatrix&) except +
    lattice_colour_matrix.LatticeColourMatrix operator+(const lattice_colour_matrix.LatticeColourMatrix&, const colour_matrix.ColourMatrix&) except +
    lattice_colour_matrix.LatticeColourMatrix operator-(const lattice_colour_matrix.LatticeColourMatrix&, const colour_matrix.ColourMatrix&) except +
    lattice_colour_matrix.LatticeColourMatrix operator*(const lattice_colour_matrix.LatticeColourMatrix&, const lattice_colour_matrix.LatticeColourMatrix&) except +
    lattice_colour_matrix.LatticeColourMatrix operator+(const lattice_colour_matrix.LatticeColourMatrix&, const lattice_colour_matrix.LatticeColourMatrix&) except +
    lattice_colour_matrix.LatticeColourMatrix operator-(const lattice_colour_matrix.LatticeColourMatrix&, const lattice_colour_matrix.LatticeColourMatrix&) except +
    gauge_field.GaugeField operator*(const lattice_colour_matrix.LatticeColourMatrix&, const gauge_field.GaugeField&) except +
    gauge_field.GaugeField operator+(const lattice_colour_matrix.LatticeColourMatrix&, const gauge_field.GaugeField&) except +
    gauge_field.GaugeField operator*(const gauge_field.GaugeField&, const colour_matrix.ColourMatrix&) except +
    gauge_field.GaugeField operator+(const gauge_field.GaugeField&, const colour_matrix.ColourMatrix&) except +
    gauge_field.GaugeField operator-(const gauge_field.GaugeField&, const colour_matrix.ColourMatrix&) except +
    gauge_field.GaugeField operator*(const gauge_field.GaugeField&, const lattice_colour_matrix.LatticeColourMatrix&) except +
    gauge_field.GaugeField operator+(const gauge_field.GaugeField&, const lattice_colour_matrix.LatticeColourMatrix&) except +
    gauge_field.GaugeField operator-(const gauge_field.GaugeField&, const lattice_colour_matrix.LatticeColourMatrix&) except +
    gauge_field.GaugeField operator*(const gauge_field.GaugeField&, const gauge_field.GaugeField&) except +
    gauge_field.GaugeField operator+(const gauge_field.GaugeField&, const gauge_field.GaugeField&) except +
    gauge_field.GaugeField operator-(const gauge_field.GaugeField&, const gauge_field.GaugeField&) except +
    colour_vector.ColourVector operator*(const colour_matrix.ColourMatrix&, const colour_vector.ColourVector&) except +
    fermion.Fermion operator*(const colour_matrix.ColourMatrix&, const fermion.Fermion&) except +
    lattice_colour_vector.LatticeColourVector operator*(const colour_matrix.ColourMatrix&, const lattice_colour_vector.LatticeColourVector&) except +
    fermion_field.FermionField operator*(const colour_matrix.ColourMatrix&, const fermion_field.FermionField&) except +
    fermion.Fermion operator*(const colour_matrix_array.ColourMatrixArray&, const colour_vector.ColourVector&) except +
    fermion.Fermion operator*(const colour_matrix_array.ColourMatrixArray&, const fermion.Fermion&) except +
    lattice_colour_vector.LatticeColourVector operator*(const lattice_colour_matrix.LatticeColourMatrix&, const colour_vector.ColourVector&) except +
    lattice_colour_vector.LatticeColourVector operator*(const lattice_colour_matrix.LatticeColourMatrix&, const lattice_colour_vector.LatticeColourVector&) except +
    fermion_field.FermionField operator*(const lattice_colour_matrix.LatticeColourMatrix&, const fermion_field.FermionField&) except +
    fermion_field.FermionField operator*(const gauge_field.GaugeField&, const colour_vector.ColourVector&) except +
    fermion_field.FermionField operator*(const gauge_field.GaugeField&, const lattice_colour_vector.LatticeColourVector&) except +
    fermion_field.FermionField operator*(const gauge_field.GaugeField&, const fermion_field.FermionField&) except +
    colour_vector.ColourVector operator+(const colour_vector.ColourVector&, const colour_vector.ColourVector&) except +
    colour_vector.ColourVector operator-(const colour_vector.ColourVector&, const colour_vector.ColourVector&) except +
    fermion.Fermion operator+(const colour_vector.ColourVector&, const fermion.Fermion&) except +
    lattice_colour_vector.LatticeColourVector operator+(const colour_vector.ColourVector&, const lattice_colour_vector.LatticeColourVector&) except +
    fermion_field.FermionField operator+(const colour_vector.ColourVector&, const fermion_field.FermionField&) except +
    fermion.Fermion operator+(const fermion.Fermion&, const colour_vector.ColourVector&) except +
    fermion.Fermion operator-(const fermion.Fermion&, const colour_vector.ColourVector&) except +
    fermion.Fermion operator+(const fermion.Fermion&, const fermion.Fermion&) except +
    fermion.Fermion operator-(const fermion.Fermion&, const fermion.Fermion&) except +
    lattice_colour_vector.LatticeColourVector operator+(const lattice_colour_vector.LatticeColourVector&, const colour_vector.ColourVector&) except +
    lattice_colour_vector.LatticeColourVector operator-(const lattice_colour_vector.LatticeColourVector&, const colour_vector.ColourVector&) except +
    lattice_colour_vector.LatticeColourVector operator+(const lattice_colour_vector.LatticeColourVector&, const lattice_colour_vector.LatticeColourVector&) except +
    lattice_colour_vector.LatticeColourVector operator-(const lattice_colour_vector.LatticeColourVector&, const lattice_colour_vector.LatticeColourVector&) except +
    fermion_field.FermionField operator+(const lattice_colour_vector.LatticeColourVector&, const fermion_field.FermionField&) except +
    fermion_field.FermionField operator+(const fermion_field.FermionField&, const colour_vector.ColourVector&) except +
    fermion_field.FermionField operator-(const fermion_field.FermionField&, const colour_vector.ColourVector&) except +
    fermion_field.FermionField operator+(const fermion_field.FermionField&, const lattice_colour_vector.LatticeColourVector&) except +
    fermion_field.FermionField operator-(const fermion_field.FermionField&, const lattice_colour_vector.LatticeColourVector&) except +
    fermion_field.FermionField operator+(const fermion_field.FermionField&, const fermion_field.FermionField&) except +
    fermion_field.FermionField operator-(const fermion_field.FermionField&, const fermion_field.FermionField&) except +
