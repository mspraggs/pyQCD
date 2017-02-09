from pyQCD.core.core cimport LatticeColourMatrix, LatticeColourVector

from fermions cimport FermionAction, WilsonFermionAction


cdef class FermionAction:

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("The FermionAction class should not be "
                                  "instantiated directly. Instantiate a "
                                  "derived class instead.")

    def apply_full(self, LatticeColourVector fermion_out,
                   LatticeColourVector fermion_in):
        self.instance.apply_full(fermion_out.instance[0],
                                 fermion_in.instance[0])

cdef class WilsonFermionAction(FermionAction):

    def __cinit__(self, float mass, LatticeColourMatrix gauge_field):
        self.instance = new _WilsonFermionAction(mass, gauge_field.instance[0])

    def __init__(self, *args, **kwargs):
        pass