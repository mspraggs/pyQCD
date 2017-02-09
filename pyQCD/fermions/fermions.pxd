from pyQCD.core.atomics cimport Real
from pyQCD.core cimport core

cdef extern from "fermions/types.hpp" namespace "pyQCD::python":
    cdef cppclass _FermionAction "pyQCD::python::FermionAction":
        _FermionAction(const Real, const core._LatticeColourMatrix&) except +
        void apply_full(core._LatticeColourVector&, const core._LatticeColourVector&)

    cdef cppclass _WilsonFermionAction "pyQCD::python::WilsonFermionAction"(_FermionAction):
        _WilsonFermionAction(const Real, const core._LatticeColourMatrix&) except +
        void apply_full(core._LatticeColourVector&, const core._LatticeColourVector&)


cdef class FermionAction:
    cdef _FermionAction* instance

cdef class WilsonFermionAction(FermionAction):
    pass