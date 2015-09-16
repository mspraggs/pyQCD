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
    colour_matrix.ColourMatrix operator+(const colour_matrix.ColourMatrix&, const colour_matrix.ColourMatrix&)
    colour_matrix_array.ColourMatrixArray operator+(const colour_matrix_array.ColourMatrixArray&, const colour_matrix_array.ColourMatrixArray&)
    lattice_colour_matrix.LatticeColourMatrix operator+(const lattice_colour_matrix.LatticeColourMatrix&, const lattice_colour_matrix.LatticeColourMatrix&)
    gauge_field.GaugeField operator+(const gauge_field.GaugeField&, const gauge_field.GaugeField&)
    colour_vector.ColourVector operator+(const colour_vector.ColourVector&, const colour_vector.ColourVector&)
    fermion.Fermion operator+(const fermion.Fermion&, const fermion.Fermion&)
    lattice_colour_vector.LatticeColourVector operator+(const lattice_colour_vector.LatticeColourVector&, const lattice_colour_vector.LatticeColourVector&)
    fermion_field.FermionField operator+(const fermion_field.FermionField&, const fermion_field.FermionField&)
    colour_matrix.ColourMatrix operator*(const colour_matrix.ColourMatrix&, const colour_matrix.ColourMatrix&)
    colour_matrix_array.ColourMatrixArray operator*(const colour_matrix.ColourMatrix&, const colour_matrix_array.ColourMatrixArray&)
    colour_vector.ColourVector operator*(const colour_matrix.ColourMatrix&, const colour_vector.ColourVector&)
    fermion.Fermion operator*(const colour_matrix.ColourMatrix&, const fermion.Fermion&)
    colour_matrix_array.ColourMatrixArray operator*(const colour_matrix_array.ColourMatrixArray&, const colour_matrix.ColourMatrix&)
    colour_matrix_array.ColourMatrixArray operator*(const colour_matrix_array.ColourMatrixArray&, const colour_matrix_array.ColourMatrixArray&)
    fermion.Fermion operator*(const colour_matrix_array.ColourMatrixArray&, const colour_vector.ColourVector&)
    fermion.Fermion operator*(const colour_matrix_array.ColourMatrixArray&, const fermion.Fermion&)
    lattice_colour_matrix.LatticeColourMatrix operator*(const lattice_colour_matrix.LatticeColourMatrix&, const lattice_colour_matrix.LatticeColourMatrix&)
    gauge_field.GaugeField operator*(const lattice_colour_matrix.LatticeColourMatrix&, const gauge_field.GaugeField&)
    lattice_colour_vector.LatticeColourVector operator*(const lattice_colour_matrix.LatticeColourMatrix&, const lattice_colour_vector.LatticeColourVector&)
    fermion_field.FermionField operator*(const lattice_colour_matrix.LatticeColourMatrix&, const fermion_field.FermionField&)
    gauge_field.GaugeField operator*(const gauge_field.GaugeField&, const lattice_colour_matrix.LatticeColourMatrix&)
    gauge_field.GaugeField operator*(const gauge_field.GaugeField&, const gauge_field.GaugeField&)
    fermion_field.FermionField operator*(const gauge_field.GaugeField&, const lattice_colour_vector.LatticeColourVector&)
    fermion_field.FermionField operator*(const gauge_field.GaugeField&, const fermion_field.FermionField&)
    colour_matrix.ColourMatrix operator-(const colour_matrix.ColourMatrix&, const colour_matrix.ColourMatrix&)
    colour_matrix_array.ColourMatrixArray operator-(const colour_matrix_array.ColourMatrixArray&, const colour_matrix_array.ColourMatrixArray&)
    lattice_colour_matrix.LatticeColourMatrix operator-(const lattice_colour_matrix.LatticeColourMatrix&, const lattice_colour_matrix.LatticeColourMatrix&)
    gauge_field.GaugeField operator-(const gauge_field.GaugeField&, const gauge_field.GaugeField&)
    colour_vector.ColourVector operator-(const colour_vector.ColourVector&, const colour_vector.ColourVector&)
    fermion.Fermion operator-(const fermion.Fermion&, const fermion.Fermion&)
    lattice_colour_vector.LatticeColourVector operator-(const lattice_colour_vector.LatticeColourVector&, const lattice_colour_vector.LatticeColourVector&)
    fermion_field.FermionField operator-(const fermion_field.FermionField&, const fermion_field.FermionField&)
