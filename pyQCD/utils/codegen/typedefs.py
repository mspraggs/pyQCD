"""This module contains the a series of classes for holding type information,
which is used to produce Cython syntax trees for the various types."""

from __future__ import absolute_import


class TypeDef(object):
    """Encapsulates type defintion and facilitates cython node generation."""

    def __init__(self, name, cname, cmodule, wrap_ptr):
        """Constructor for TypeDef object, See help(TypeDef)."""
        self.name = name
        self.cname = cname
        self.cmodule = cmodule
        self.wrap_ptr = wrap_ptr


class ContainerDef(TypeDef):
    """Encapsulates container definition and facilitates cython node generation.
    """

    def __init__(self, name, cname, cmodule, size_expr, shape_expr, ndims_expr,
                 buffer_ndims, element_type):
        """Constructor for ContainerDef object. See help(ContainerDef)"""
        super(ContainerDef, self).__init__(name, cname, cmodule, True)
        self.element_type = element_type
        self.size_expr = size_expr
        self.ndims_expr = ndims_expr
        self.structure = [self.__class__.__name__.replace("Def", "")]
        if isinstance(element_type, ContainerDef):
            self.structure.extend(element_type.structure)
        try:
            self.buffer_ndims = element_type.buffer_ndims + buffer_ndims
        except AttributeError:
            self.buffer_ndims = buffer_ndims
        try:
            self.shape_expr = "{} + {}".format(shape_expr,
                                               element_type.shape_expr)
        except AttributeError:
            self.shape_expr = shape_expr

        try:
            int(size_expr)
        except ValueError:
            self.is_static = False
        else:
            self.is_static = True

    @property
    def accessor_info(self):
        """Generates a list of tuples of lengths and types for accessor checks
        """
        out = [(self.ndims_expr, self.element_type)]
        try:
            out.extend(self.element_type.accessor_info)
        except AttributeError:
            pass
        return out


class MatrixDef(ContainerDef):
    """Specialise container definition for matrix type"""

    def __init__(self, name, cname, cmodule, shape, element_type):
        """Constructor for MatrixDef object. See help(MatrixDef)"""
        size = reduce(lambda x, y: x * y, shape)
        super(MatrixDef, self).__init__(name, cname, cmodule, str(size),
                                        str(shape), str(len(shape)),
                                        len(shape), element_type)
        self.shape = shape
        self.is_matrix = len(self.shape) == 2
        self.is_square = self.is_matrix and self.shape[0] == self.shape[1]


class ArrayDef(ContainerDef):
    """Specialise container definition for array type"""

    def __init__(self, name, cname, cmodule, element_type):
        """Constructor for ArrayDef object. See help(ArrayDef)."""
        super(ArrayDef, self).__init__(name, cname, cmodule,
                                        "self.instance.size()", "(1,)", "1", 1,
                                        element_type)


class LatticeDef(ContainerDef):
    """Specialise container definition for lattice type"""

    def __init__(self, name, cname, cmodule, element_type):
        """Constructor for LatticeDef object. See help(LatticeDef)"""
        super(LatticeDef, self).__init__(name, cname, cmodule,
                                         "self.instance.volume()",
                                         "tuple(self.instance.lattice_shape())",
                                         "self.instance.num_dims()", 1,
                                         element_type)