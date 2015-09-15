"""This module contains template tags to generate code snippets for the various
pxd and C++ header templates"""


def cpptype(typedef):
    """Create C++ type declaration for the supplied type definition.

    Args:
      typedef (ContainerDef): TypeDef instance specifying the nested type
        declaration.
      precision (str): The fundamental machine type for representing real
        numbers.
    """

    if typedef.structure[0] == "Matrix":
        shape = typedef.shape + ((1,) if len(typedef.shape) == 1 else ())
        template = "Eigen::Matrix<Complex, {}, {}>"
    elif typedef.structure[0] == "Array":
        shape = (typedef.element_type.shape +
                 ((1,) if len(typedef.element_type.shape) == 1 else ()))
        template = "pyQCD::MatrixArray<{}, {}, Real>"
    elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Matrix":
        shape = (typedef.element_type.shape +
                 ((1,) if len(typedef.element_type.shape) == 1 else ()))
        template = ("pyQCD::Lattice<Eigen::Matrix<Complex, {}, {}>, "
                    "Eigen::aligned_allocator>")
    elif typedef.structure[0] == "Lattice" and typedef.structure[1] == "Array":
        shape = (typedef.element_type.element_type.shape +
                 ((1,) if len(typedef.element_type.element_type.shape) == 1
                  else ()))
        template = "pyQCD::Lattice<pyQCD::MatrixArray<{}, {}, Real> >"
    else:
        raise ValueError("Supplied typedef structure not recognised: {}"
                         .format(typedef.structure))
    return template.format(*shape)
