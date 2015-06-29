"""This module contains the a series of classes for holding type information,
which is used to produce Cython syntax trees for the various types."""

from __future__ import absolute_import

from . import nodegen


class TypeDef(object):
    """Encapsulates type defintion and facilitates cython node generation."""

    def __init__(self, name, cname):
        """Constructor for TypeDef object, See help(TypeDef)."""
        self.name = name
        self.cname = cname


class ContainerDef(TypeDef):
    """Encapsulates container definition and facilitates cython node generation.
    """

    def __init__(self, name, cname, num_dims, element_type=None):
        """Constructor for ContainerDef object. See help(ContainerDef)"""
        super(ContainerDef, self).__init__(name, cname)
        self.element_type = element_type
        self.num_dims = num_dims


class MatrixDef(ContainerDef):
    """Specialise container definition for matrix type"""

    def __init__(self, name, cname, element_type, num_rows, num_cols):
        """Constructor for MatrixDef object. See help(MatrixDef)"""
        super(MatrixDef, self).__init__(name, cname, element_type)
        self.num_rows = num_rows
        self.num_cols = num_cols