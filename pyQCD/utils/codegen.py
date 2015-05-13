"""This module contains various utility functions. Specifically:

- Functions to generate Cython code a specific number of colours, representation
etc.
"""

import os
from collections import namedtuple
from itertools import product
from string import lowercase

from jinja2 import Environment, PackageLoader, FileSystemLoader
from pkg_resources import Requirement, resource_filename


# Create the jinja2 template environment. If we're installed, then look for
# templates in the packages installation tree. Otherwise, look for it in the
# root of the project source tree.th
env = Environment(loader=PackageLoader('pyQCD', 'templates'),
                  trim_blocks=True, lstrip_blocks=True)

MatrixDefinition = namedtuple("MatrixDefinition",
                              ["num_rows", "num_cols", "matrix_name",
                               "array_name", "lattice_matrix_name",
                               "lattice_array_name"])


def _camel2underscores(string):
    """Converts a string in titlecase or camelcase to underscores"""
    for char in lowercase:
        string = string.replace(char.upper(), '_' + char)
    return string.lstrip('_')


def create_matrix_definition(num_rows, num_cols, matrix_name, array_name=None,
                             lattice_matrix_name=None, lattice_array_name=None):
    """Create a MatrixDefinition namedtuple using the supplied arguments.

    This function sets up default values for array_name and lattice_name
    if necessary.

    Args:
      num_rows (int): The number of rows required in the matrix.
      num_cols (int): The number of columns required in the matrix.
      matrix_name (str): The fundamental type name of the matrix, which is used
        to refer to this matrix type throughout any Cython code.
      array_name (str, optional): The type name given to an array of such matrix
        objects. Defaults to {matrix_name}Array.
      lattice_matrix_name (str, optional): The type name given to a lattice of
        these matrix objects. Defaults to Lattice{matrix_name}.
      lattice_array_name (str, optional): The type name given to a lattice of
        arrays of these matrix objects. Defaults to Lattice{matrix_name}Array.

    Returns:
      MatrixDefinition: A named tuple containing the supplied parameters.
    """

    return MatrixDefinition(
        num_rows, num_cols, matrix_name,
        array_name or "{}Array".format(matrix_name),
        lattice_matrix_name or "Lattice{}".format(matrix_name),
        lattice_array_name or "Lattice{}Array".format(matrix_name))


def generate_cython_types(output_path, precision, matrices):
    """Generate Cython matrix, array and lattice types.

    This function gathers all jinja2 templates in the package templates
    directory and

    Args:
      output_path (str): The output directory in which to put the generated
        code.
      precision (str): The fundamental machine type to be used throughout the
        code (e.g. 'double' or 'float').
      matrices (iterable): An iterable object containing instances of
        MatrixDefinition.
    """

    variants = ['matrix', 'array', 'lattice_matrix', 'lattice_array']
    # List of tuples of allowed binary operations
    scalar_binary_ops = []
    matrix_shapes = [(m.num_rows, m.num_cols) for m in matrices]
    operator_includes = []

    for matrix in matrices:
        fnames = [_camel2underscores(getattr(matrix, "{}_name".format(variant)))
                  for variant in variants]
        includes = dict([("{}_include".format(variant), fname)
                         for variant, fname in zip(variants, fnames)])
        for variant, fname in zip(variants, fnames):
            name = getattr(matrix, "{}_name".format(variant))
            template = env.get_template("core/{}.pxd".format(variant))
            operator_includes.append((fname, name))
            print("Writing {} to {}".format(variant, fname))
            with open(os.path.join(output_path, fname + ".pxd"), 'w') as f:
                f.write(template.render(precision=precision, matrixdef=matrix,
                                        includes=includes))
            # Add scalar binary operations to the list of possible operators
            for op in '+*-':
                scalar_binary_ops.extend([
                    (name, op, "Real", name), (name, op, name, "Real"),
                    (name, op, "Complex", name), (name, op, name, "Complex")
                ])
            scalar_binary_ops.extend([
                (name, '/', name, "Real"), (name, '/', name, "Complex")
            ])

    broadcast_ops = []
    non_broadcast_ops = []

    ret_lookup = dict(zip(matrix_shapes, matrices))

    for matrix_lhs, matrix_rhs in product(matrices, matrices):
        # Check that the operation is allowed
        if (matrix_lhs.num_cols == matrix_rhs.num_rows
            and (matrix_lhs.num_rows, matrix_rhs.num_cols) in matrix_shapes):

            for variant_lhs, variant_rhs in product(variants, variants):
                name_lhs = getattr(matrix_lhs, "{}_name".format(variant_lhs))
                name_rhs = getattr(matrix_rhs, "{}_name".format(variant_rhs))
                # Now we need to check whether we need to broadcast an array
                # against a lattice type.
                array_lhs = 'array' in variant_lhs
                array_rhs = 'array' in variant_rhs
                lattice_lhs = 'lattice' in variant_lhs
                lattice_rhs = 'lattice' in variant_rhs
                lhs = {'name': name_lhs,
                       'broadcast': lattice_rhs and not lattice_lhs}
                rhs = {'name': name_rhs,
                       'broadcast': lattice_lhs and not lattice_rhs}
                ret = ret_lookup[matrix_lhs.num_rows, matrix_rhs.num_cols]
                ret_array = array_lhs or array_rhs
                ret_lattice = lattice_lhs or lattice_rhs
                ret_variant = (('lattice_' if ret_lattice else '') +
                               ('array' if ret_array else 'matrix'))
                ret_name = getattr(ret, "{}_name".format(ret_variant))
                for op in '*+-':
                    if lhs['broadcast'] or rhs['broadcast']:
                        broadcast_ops.append((ret_name, op, lhs, rhs))
                    else:
                        non_broadcast_ops.append((ret_name, op, lhs, rhs))

    types_template = env.get_template("core/types.hpp")
    with open(os.path.join(output_path, "types.hpp"), 'w') as f:
        f.write(types_template.render(matrices=matrices, precision=precision))

    cython_operator_template = env.get_template("core/operators.pxd")
    with open(os.path.join(output_path, "operators.pxd"), 'w') as f:
        f.write(cython_operator_template.render(
            scalar_binary_ops=scalar_binary_ops,
            non_broadcast_binary_ops=non_broadcast_ops,
            broadcast_binary_ops=broadcast_ops,
            includes=operator_includes))
    # Here we generate some C++ code to wrap operators where one of the operands
    # is an array type and the other a lattice type.
    cpp_operator_template = env.get_template("core/broadcast_operators.hpp")
    with open(os.path.join(output_path, "broadcast_operators.hpp"), 'w') as f:
        f.write(cpp_operator_template.render(ops=broadcast_ops))


env.filters['to_underscores'] = _camel2underscores