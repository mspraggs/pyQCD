"""This module contains the a series of classes for holding type information,
which is used to produce Cython syntax trees for the various types."""

from __future__ import absolute_import


class TypeDef(object):
    """Encapsulates type defintion and facilitates cython node generation."""

    def __init__(self, name, cname, cmodule):
        """Constructor for TypeDef object, See help(TypeDef)."""
        self.name = name
        self.cname = cname
        self.cmodule = cmodule
        

class ContainerDef(TypeDef):
    """Encapsulates container definition and facilitates cython node generation.
    """

    def __init__(self, name, cname, cmodule, element_type, def_template,
                 impl_template):
        """Constructor for ContainerDef object. See help(ContainerDef)"""
        super(ContainerDef, self).__init__(name, cname, cmodule)
        self.element_type = element_type
        self.def_template = def_template
        self.impl_template = impl_template

class MatrixDef(ContainerDef):
    """Specialise container definition for matrix type"""

    def __init__(self, name, cname, cmodule, shape, element_type):
        """Constructor for MatrixDef object. See help(MatrixDef)"""
        size = reduce(lambda x, y: x * y, shape)
        super(MatrixDef, self).__init__(name, cname, cmodule, element_type,
                                        "core/matrix.pxd", "core/matrix.pyx")
        self.shape = shape
        self.size = size
        self.ndims = len(self.shape)


class LatticeDef(ContainerDef):
    """Specialise container definition for lattice type"""

    def __init__(self, name, cname, cmodule, element_type):
        """Constructor for LatticeDef object. See help(LatticeDef)"""
        super(LatticeDef, self).__init__(name, cname, cmodule, element_type,
                                         "core/lattice.pxd", "core/lattice.pyx")

