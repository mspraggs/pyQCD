from pyQCD.core.core cimport LatticeColourMatrix, LatticeColourVector

from fermions cimport FermionAction, WilsonFermionAction


cdef class FermionAction:

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("The FermionAction class should not be "
                                  "instantiated directly. Instantiate a "
                                  "derived class instead.")

    def apply_full(self, LatticeColourVector fermion_in):
        fermion_out = LatticeColourVector(fermion_in.layout, fermion_in.site_size)
        fermion_out.instance[0] = self.instance.apply_full(
            fermion_in.instance[0])
        return fermion_out

cdef class WilsonFermionAction(FermionAction):

    def __cinit__(self, float mass, LatticeColourMatrix gauge_field,
                  boundary_phase_angles):
        self.instance = new _WilsonFermionAction(mass, gauge_field.instance[0],
                                                 list(boundary_phase_angles))

    def __init__(self, *args, **kwargs):
        pass