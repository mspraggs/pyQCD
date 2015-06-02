from fermion cimport Fermion
from layout cimport Layout


cdef extern from "types.hpp":
    cdef cppclass FermionField:
        FermionField() except +
        FermionField(const Layout&, const Fermion) except +
        unsigned int volume()