"""This module contains functions for generating attribute and member function
code for each of the core types in core.pyx"""


def allocation_code(typedef):
    """Generate code for constructors and destructors.

    Args:
      typedef (ContainerDef): A ContainerDef instance specifying the type to
        generate code for.
    """
    return ""


def setget_code(typedef):
    """Generate code for __setitem__ and __getitem__ member functions.

    Args:
      typedef (ContainerDef): A ContainerDef instance specifying the type to
        generate code for.
    """
    return ""


def buffer_code(typedef):
    """Generate code for __getbuffer__ and __releasebuffer__ member functions.

    Args:
      typedef (ContainerDef): A ContainerDef instance specifying the type to
        generate code for.
    """
    return ""


def static_func_code(typedef):
    """Generate code for zeros, ones, identity and similar static initialisers.

    Args:
      typedef (ContainerDef): A ContainerDef instance specifying the type to
        generate code for.
    """
    return ""