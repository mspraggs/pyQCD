"""This module contains functions for generating attribute and member function
code for each of the core types in core.pyx"""


def allocation_code(typedef):
    """Generate code for constructors and destructors.

    Args:
      typedef (ContainerDef): A ContainerDef instance specifying the type to
        generate code for.
    """
    from . import env
    template = env.get_template("core/allocation.pyx")
    return template.render(typedef=typedef)


def setget_code(typedef, precision):
    """Generate code for __setitem__ and __getitem__ member functions.

    Args:
      typedef (ContainerDef): A ContainerDef instance specifying the type to
        generate code for.
    """
    from . import env
    template = env.get_template("core/setget.pyx")
    return template.render(typedef=typedef, precision=precision)


def buffer_code(typedef, precision):
    """Generate code for __getbuffer__ and __releasebuffer__ member functions.

    Args:
      typedef (ContainerDef): A ContainerDef instance specifying the type to
        generate code for.
    """
    from . import env
    template = env.get_template("core/buffer.pyx")
    shape = typedef.matrix_shape
    if len(shape) > 1:
        num_rows, num_cols = shape
    else:
        num_rows, num_cols = shape[0], 1
    return template.render(typedef=typedef, precision=precision,
                           matrix_size=reduce(lambda x, y: x * y, shape, 1),
                           num_rows=num_rows, num_cols=num_cols,
                           is_matrix=num_rows == num_cols)


def static_func_code(typedef):
    """Generate code for zeros, ones, identity and similar static initialisers.

    Args:
      typedef (ContainerDef): A ContainerDef instance specifying the type to
        generate code for.
    """
    from . import env
    template = env.get_template("core/member_funcs.pyx")
    return template.render(typedef=typedef)