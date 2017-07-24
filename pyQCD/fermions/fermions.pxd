from libcpp.vector cimport vector

from pyQCD.core.atomics cimport Real
from pyQCD.core cimport core

cdef extern from "fermions/fermion_action.hpp" namespace "pyQCD::fermions":
    cdef cppclass _FermionAction "pyQCD::fermions::Action<pyQCD::Real, pyQCD::num_colours>":
        _FermionAction(const Real, const vector[Real]&) except +
        core._LatticeColourVector apply_full(const core._LatticeColourVector&)

cdef extern from "fermions/wilson_action.hpp" namespace "pyQCD::fermions":
    cdef cppclass _WilsonFermionAction "pyQCD::fermions::WilsonAction<pyQCD::Real, pyQCD::num_colours>"(_FermionAction):
        _WilsonFermionAction(const Real, const core._LatticeColourMatrix&, const vector[Real]&) except +
        core._LatticeColourVector apply_full(const core._LatticeColourVector&)


cdef class FermionAction:
    cdef _FermionAction* instance

cdef class WilsonFermionAction(FermionAction):
    pass