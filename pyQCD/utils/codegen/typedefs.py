"""This module contains the a series of classes for holding type information,
which is used to produce Cython syntax trees for the various types."""

from __future__ import absolute_import

from Cython.Compiler import ExprNodes

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

    def __init__(self, name, cname, ndims_expr, size_expr, element_type=None):
        """Constructor for ContainerDef object. See help(ContainerDef)"""
        super(ContainerDef, self).__init__(name, cname)
        self.element_type = element_type
        self.ndims_expr = ndims_expr
        self.size_expr = size_expr
        self.is_static = isinstance(size_expr, ExprNodes.IntNode)

    @property
    def buffer_ndims(self):
        """Calculate the number of dimensions a buffer object must use"""
        self_ndims = (int(self.ndims_expr.value)
                      if isinstance(self.ndims_expr, ExprNodes.IntNode)
                      else 1)
        try:
            child_ndims = self.element_type.buffer_ndims
        except AttributeError:
            child_ndims = 0
        return self_ndims + child_ndims


class MatrixDef(ContainerDef):
    """Specialise container definition for matrix type"""

    def __init__(self, name, cname, shape, element_type=None):
        """Constructor for MatrixDef object. See help(MatrixDef)"""
        size = reduce(lambda x, y: x * y, shape)
        ndims_expr = ExprNodes.IntNode(None, value=str(len(shape)))
        size_expr = ExprNodes.IntNode(None, value=str(size))
        super(MatrixDef, self).__init__(name, cname, ndims_expr, size_expr,
                                        element_type)
        self.shape = shape