from pyQCD.core.core cimport LatticeColourMatrix, Layout, LexicoLayout

from gauge cimport GaugeAction, WilsonGaugeAction, SymanzikGaugeAction, _average_plaquette


cdef class GaugeAction:

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("The GaugeAction class should not be "
                                  "instantiated directly. Instantiate a "
                                  "derived class instead.")

cdef class WilsonGaugeAction:

    def __cinit__(self, float beta, shape):
        cdef const Layout* layout = new LexicoLayout(shape)
        self.instance = new _WilsonGaugeAction(beta, layout[0])

    def __init__(self, *args, **kwargs):
        pass

cdef class SymanzikGaugeAction:

    def __cinit__(self, float beta, shape):
        cdef const Layout* layout = new LexicoLayout(shape)
        self.instance = new _RectangleGaugeAction(beta, layout[0], -1.0 / 12.0)

    def __init__(self, *args, **kwargs):
        pass

cdef class IwasakiGaugeAction:

    def __cinit__(self, float beta, shape):
        cdef const Layout* layout = new LexicoLayout(shape)
        self.instance = new _RectangleGaugeAction(beta, layout[0], -0.331)

    def __init__(self, *args, **kwargs):
        pass

def average_plaquette(LatticeColourMatrix gauge_field):
    return _average_plaquette(gauge_field.instance[0])

def average_rectangle(LatticeColourMatrix gauge_field):
    return _average_rectangle(gauge_field.instance[0])