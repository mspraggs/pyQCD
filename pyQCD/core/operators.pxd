cimport complex
cimport colour_matrix
cimport lattice_colour_matrix
cimport colour_vector
cimport lattice_colour_vector


cdef extern from "types.hpp":
    colour_matrix.ColourMatrix operator+(const colour_matrix.ColourMatrix&, const colour_matrix.ColourMatrix&)
    lattice_colour_matrix.LatticeColourMatrix operator+(const lattice_colour_matrix.LatticeColourMatrix&, const lattice_colour_matrix.LatticeColourMatrix&)
    colour_vector.ColourVector operator+(const colour_vector.ColourVector&, const colour_vector.ColourVector&)
    lattice_colour_vector.LatticeColourVector operator+(const lattice_colour_vector.LatticeColourVector&, const lattice_colour_vector.LatticeColourVector&)
    colour_matrix.ColourMatrix operator*(const colour_matrix.ColourMatrix&, const colour_matrix.ColourMatrix&)
    lattice_colour_matrix.LatticeColourMatrix operator*(const colour_matrix.ColourMatrix&, const lattice_colour_matrix.LatticeColourMatrix&)
    colour_vector.ColourVector operator*(const colour_matrix.ColourMatrix&, const colour_vector.ColourVector&)
    lattice_colour_vector.LatticeColourVector operator*(const colour_matrix.ColourMatrix&, const lattice_colour_vector.LatticeColourVector&)
    lattice_colour_matrix.LatticeColourMatrix operator*(const lattice_colour_matrix.LatticeColourMatrix&, const colour_matrix.ColourMatrix&)
    lattice_colour_matrix.LatticeColourMatrix operator*(const lattice_colour_matrix.LatticeColourMatrix&, const lattice_colour_matrix.LatticeColourMatrix&)
    lattice_colour_vector.LatticeColourVector operator*(const lattice_colour_matrix.LatticeColourMatrix&, const colour_vector.ColourVector&)
    lattice_colour_vector.LatticeColourVector operator*(const lattice_colour_matrix.LatticeColourMatrix&, const lattice_colour_vector.LatticeColourVector&)
    colour_matrix.ColourMatrix operator-(const colour_matrix.ColourMatrix&, const colour_matrix.ColourMatrix&)
    lattice_colour_matrix.LatticeColourMatrix operator-(const lattice_colour_matrix.LatticeColourMatrix&, const lattice_colour_matrix.LatticeColourMatrix&)
    colour_vector.ColourVector operator-(const colour_vector.ColourVector&, const colour_vector.ColourVector&)
    lattice_colour_vector.LatticeColourVector operator-(const lattice_colour_vector.LatticeColourVector&, const lattice_colour_vector.LatticeColourVector&)
