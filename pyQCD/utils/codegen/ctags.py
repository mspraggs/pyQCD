
def cpptype(typedef, precision):
    """Create C++ type declaration for the supplied type definition.

    Args:
      typedef (ContainerDef): TypeDef instance specifying the nested type
        declaration.
    """

    if typedef.structure[0] == "Matrix":
        shape = typedef.shape + ((1,) if len(typedef.shape) == 1 else ())
        return "Eigen::Matrix<{}, {}, {}>".format(typedef.element_type.cname,
                                                  *shape)
    elif typedef.structure[0] == "Array":
        shape = (typedef.element_type.shape +
                 ((1,) if len(typedef.element_type.shape) == 1 else ()))
        return "pyQCD::MatrixArray<{}, {}, {}>".format(shape[0], shape[1],
                                                       precision)
    elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Matrix":
        shape = (typedef.element_type.shape +
                 ((1,) if len(typedef.element_type.shape) == 1 else ()))
        return ("pyQCD::Lattice<Eigen::Matrix<{}, {}, {}>, "
                "Eigen::aligned_allocator>"
                .format(typedef.element_type.element_type.cname, *shape))
    elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Array":
        shape = (typedef.element_type.element_type.shape +
                 ((1,) if len(typedef.element_type.element_type.shape) == 1
                  else ()))
        return ("pyQCD::Lattice<pyQCD::MatrixArray<{}, {}, {}> >"
                .format(shape[0], shape[1], precision))
    else:
        raise ValueError("Supplied typedef structure not recognised: {}"
                         .format(typedef.structure))