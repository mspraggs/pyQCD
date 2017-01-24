"""This module contains the a series of classes for holding type information,
which is used to produce Cython syntax trees for the various types."""

from __future__ import absolute_import


class TypeDef(object):
    """Encapsulates type defintion and facilitates cython node generation."""

    def __init__(self, name, cname, cmodule, wrap_ptr, builtin=False):
        """Constructor for TypeDef object, See help(TypeDef)."""
        self.name = name
        self.cname = cname
        self.cmodule = cmodule
        self.wrap_ptr = wrap_ptr
        self.builtin = builtin
        self.cmembers = []
        self.ctor_args = []

    def add_cmember(self, typename, varname, init=None):
        """Add a statically typed member variable to this type definition"""
        self.cmembers.append((typename, varname, init))

    def add_ctor_arg(self, varname, typename=None, default=None):
        """Add an argument to the type's constructor"""
        self.ctor_args.append((typename, varname, default))

    @property
    def ctor_argstring(self):
        """Create the argument string for this type's constructor"""
        arglist = []
        for typename, varname, default in self.ctor_args:
            arg_substring = "{} ".format(typename) if typename else ""
            arg_substring += varname
            arg_substring += "={}".format(default) if default else ""
            arglist.append(arg_substring)
        return ", ".join(arglist)

    @property
    def cpptype(self):
        """C++ string specifying the type"""
        raise NotImplementedError

    def accessor(self, varname, broadcast=False):
        if self.builtin:
            return varname
        else:
            return "{}.instance{}{}".format(varname,
                                            "[0]" if self.wrap_ptr else "",
                                            ".broadcast()" if broadcast else "")


class ContainerDef(TypeDef):
    """Encapsulates container definition and facilitates cython node generation.
    """

    def __init__(self, name, cname, cmodule, size_expr, shape_expr,
                 buffer_ndims, element_type, init_template):
        """Constructor for ContainerDef object. See help(ContainerDef)"""
        super(ContainerDef, self).__init__(name, cname, cmodule, True)
        self.element_type = element_type
        self.size_expr = size_expr
        self._shape_expr = shape_expr
        self.structure = [self.__class__.__name__.replace("Def", "")]
        self.init_template = init_template
        if isinstance(element_type, ContainerDef):
            self.structure.extend(element_type.structure)
        try:
            self.buffer_ndims = element_type.buffer_ndims + buffer_ndims
        except AttributeError:
            self.buffer_ndims = buffer_ndims
        try:
            int(size_expr)
        except ValueError:
            self.is_static = False
        else:
            self.is_static = True

    def get_lattice_name(self, other, self_name, other_name):
        """Returns one of the supplied names if self or other are LatticeDefs

        This function is used to retrieve a name to extract a shape attribute
        from one of the operands in an arithmetic operation.

        Args:
          other (ContainerDef): TypeDef for the second operand in the operation.
          self_name (str): String giving the name that refers to an instance of
            the type defined by self.
          other_name (str): String giving the name that refers to an instance of
            the type defined by other.

        Returns:
          str or NoneType: self_name, other_name or None.

          This function will return the first name that is associated with a
          type described by a LatticeDef object. If neither self nor other are,
          then None is returned.
        """

        if isinstance(self, LatticeDef):
            return self_name
        elif isinstance(other, LatticeDef):
            return other_name
        else:
            return None

    @property
    def matrix_shape(self):
        """The shape of the root child element type, if it exists"""
        if isinstance(self, MatrixDef):
            return self.shape
        else:
            return self.element_type.matrix_shape

    def unpack(self):
        """Returns a list of TypeDef instances"""
        out = [self]
        try:
            out.extend(self.element_type.unpack())
        except AttributeError:
            pass
        return out

    @property
    def shape_expr(self):
        """Return expression that corresponds to the shape of the container.
        Used in the reshaping of numpy buffer."""
        try:
            return "{} + {}".format(self._shape_expr,
                                    self.element_type.shape_expr)
        except AttributeError:
            return self._shape_expr

    @property
    def init_code(self):
        """Call to the underlying C++ constructor for this type"""
        try:
            return self.init_template.format(self.element_type.init_code)
        except AttributeError:
            return self.init_template

    def buffer_info(self, item_size_expr):
        """Generates information required to construct a __buffer__ function

        Args:
          item_size_expr (str): The variable name defining, as a number of
            bytes, the smallest possible stride that can be made along the
            buffer.

        Returns:
          tuple: Containing two elements, themselves defining the buffer.

          The first element is a list of two-tuples, which define the extent
          and stride length in bytes for the various view axes. The second
          element is an expression for the total size of the buffer, in bytes.
        """

        inst_ref = self.accessor("self")
        stride_length = item_size_expr
        buffer_iter = []
        types = self.unpack()
        it = enumerate(types)

        for depth, tdef in reversed(list(it)):
            if type(tdef) is MatrixDef:
                buffer_iter.append((stride_length, tdef.shape[0]))
                if len(tdef.shape) > 1:
                    info = (stride_length + " * " + str(tdef.shape[1]),
                            tdef.shape[1])
                    buffer_iter.insert(0, info)
                stride_length += " * {}".format(tdef.size_expr)
            else:
                size_expr = inst_ref + "[0]" * depth + "." + tdef.size_expr
                buffer_iter.append((stride_length, size_expr))
                stride_length += " * {}".format(size_expr)

        return buffer_iter[::-1], stride_length

    def _can_mutliply(self, typedef):
        """Based on matrix_shape, checks whether typedef can left-multiply self.
        """
        try:
            return self.matrix_shape[1] == typedef.matrix_shape[0]
        except IndexError:
            return False

    def _result_shape(self, typedef):
        """Generates the resulting matrix_shape from performing self * typedef.
        """
        try:
            return self.matrix_shape[0], typedef.matrix_shape[1]
        except IndexError:
            return self.matrix_shape[0],

    def generate_arithmetic_operations(self, typedefs, operations=None):
        """Generate possible arithmetic operations from a list of ContainerDefs

        Args:
          typedefs (list): A list of ContainerDef instances to compare this
            typedef against and then, if possible, specify arithmetic operations
            for.
          typedefs (dict): Pre-existing result dict to add operations into.

        Returns:
          dict: Maps operations to lists of operation arguments/return values.

          Each list is a tuple with three elements. The first is the TypeDef of
          the result of the operation. The second and third are the TypeDefs of
          the first and second operands, respectively.
        """

        operations = operations or {k: [] for k in "*/+-"}

        lhs_is_lattice = isinstance(self, LatticeDef)

        def filter_types(typedef, shape, is_lattice):
            """Checks whether typedef matches criteria (shape and type)"""
            return (typedef.matrix_shape == shape and
                    isinstance(typedef, LatticeDef) == is_lattice)

        for other in typedefs:
            rhs_is_lattice = isinstance(other, LatticeDef)
            result_is_lattice = lhs_is_lattice or rhs_is_lattice
            result_shape = self._result_shape(other)
            filter_args = result_shape, result_is_lattice
            try:
                result_typedef = [e for e in typedefs
                                  if filter_types(e, *filter_args)][0]
            except IndexError:
                continue
            can_multiply = self._can_mutliply(other)
            can_addsub = (self.matrix_shape == other.matrix_shape and
                          rhs_is_lattice == lhs_is_lattice)

            if can_multiply:
                operations["*"].append((result_typedef, self, other))
            if can_addsub:
                for op in "+-":
                    operations[op].append((result_typedef, self, other))

        return operations


class MatrixDef(ContainerDef):
    """Specialise container definition for matrix type"""

    def __init__(self, name, cname, cmodule, shape, element_type):
        """Constructor for MatrixDef object. See help(MatrixDef)"""
        size = reduce(lambda x, y: x * y, shape)
        super(MatrixDef, self).__init__(
            name, cname, cmodule, str(size), str(shape), len(shape),
            element_type, "{}.{}({}.zeros())".format(cmodule, cname, cmodule))
        self.shape = shape
        self.is_matrix = len(self.shape) == 2
        self.is_square = self.is_matrix and self.shape[0] == self.shape[1]
        self.add_cmember("int", "view_count", "0")
        self.add_cmember("Py_ssize_t", "buffer_shape[{}]".format(len(shape)))
        self.add_cmember("Py_ssize_t", "buffer_strides[{}]".format(len(shape)))

        self.def_template = "core/matrix.pxd"
        self.impl_template = "core/matrix.pyx"

    @property
    def cpptype(self):
        num_colours = self.matrix_shape[0]
        if self.is_matrix:
            return "ColourMatrix<Real, {}>".format(num_colours)
        else:
            return "ColourVector<Real, {}>".format(num_colours)


class LatticeDef(ContainerDef):
    """Specialise container definition for lattice type"""

    def __init__(self, name, cname, cmodule, element_type):
        """Constructor for LatticeDef object. See help(LatticeDef)"""
        shape_expr = "tuple(self.lexico_layout.shape()) + (self.site_size,)"
        super(LatticeDef, self).__init__(
            name, cname, cmodule, "volume() * self.site_size", shape_expr, 1,
            element_type, "{}.{}(self.lexico_layout[0], {{}}, site_size)"
            .format(cmodule, cname))

        shape = element_type.matrix_shape
        self.add_ctor_arg("shape")
        self.add_ctor_arg("site_size", "int", "1")
        self.add_cmember("layout.Layout*", "lexico_layout",
                         "new layout.LexicoLayout(shape)")
        self.add_cmember("int", "view_count", "0")
        self.add_cmember("int", "site_size", "site_size")
        self.add_cmember("Py_ssize_t",
                         "buffer_shape[{}]".format(len(shape) + 1))
        self.add_cmember("Py_ssize_t",
                         "buffer_strides[{}]".format(len(shape) + 1))

        self.def_template = "core/lattice.pxd"
        self.impl_template = "core/lattice.pyx"

    @property
    def cpptype(self):
        num_colours = self.matrix_shape[0]
        if len(self.matrix_shape) > 1:
            return "LatticeColourMatrix<Real, {}>".format(num_colours)
        else:
            return "LatticeColourVector<Real, {}>".format(num_colours)