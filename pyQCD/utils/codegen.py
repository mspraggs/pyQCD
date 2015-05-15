"""This module contains various utility functions. Specifically:

- Functions to generate Cython code a specific number of colours, representation
etc.
"""

from __future__ import absolute_import, print_function

import os
from collections import namedtuple
from itertools import product
from string import lowercase

from jinja2 import Environment, PackageLoader


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


def get_compatible_variants(matrix_lhs, matrix_rhs):
    """Returns triplets of compatible variants and return variants

    Args:
      matrix_lhs (MatrixDefinition): The matrix on the left hand side of the
        binary operator.
      matrix_rhs (MatrixDefinition): The matrix on the right hand side of the
        binary operator.

    Returns:
      list: List of three-tuples containing compatible types and return types
    """

    if matrix_lhs.num_cols != matrix_rhs.num_rows:
        return (-1, -1), []
    else:
        result_shape = (matrix_lhs.num_rows, matrix_rhs.num_cols)

    variants = ['matrix', 'array', 'lattice_matrix', 'lattice_array']
    pairs = []
    for variant_lhs, variant_rhs in product(variants, variants):
        lattice_lhs = 'lattice' in variant_lhs
        lattice_rhs = 'lattice' in variant_rhs
        ret_array = 'array' in variant_lhs or 'array' in variant_rhs
        ret_lattice = lattice_lhs or lattice_rhs
        ret_variant = (("lattice_" if ret_lattice else "") +
                       ("array" if ret_array else "matrix"))
        if lattice_lhs == lattice_rhs:
            pairs.append((variant_lhs, variant_rhs, ret_variant))
        elif variant_lhs == 'matrix' or variant_rhs == 'matrix':
            pairs.append((variant_lhs, variant_rhs, ret_variant))

    return result_shape, pairs


def make_lattice_binary_ops(matrices, matrix_lhs, matrix_rhs):
    """Create a list of tuples that define possible lattice binary operators

    Args:
      matrices (list): A list of MatrixDefinition instances.
      matrix_lhs (MatrixDefinition): The matrix on the lhs of the operation.
      matrix_rhs (MatrixDefinition): The matrix on the rhs of the operation.

    Returns:
      list: List of four-tuples containing return value, operands and operator
    """
    ops = []
    ret_shape, variant_triplets = get_compatible_variants(matrix_lhs,
                                                          matrix_rhs)
    ret_lookup = dict([((m.num_rows, m.num_cols), m) for m in matrices])
    try:
        matrix_ret = ret_lookup[ret_shape]
    except KeyError:
        return []
    for variant_triplet in variant_triplets:
        lhs_name, rhs_name, ret_name = tuple([
            "{}_name".format(var) for var in variant_triplet
        ])
        for op in '*+-':
            ops.append((getattr(matrix_ret, ret_name), op,
                        getattr(matrix_lhs, lhs_name),
                        getattr(matrix_rhs, rhs_name)))

    return ops


def make_scalar_binary_ops(matrix):
    """Create a list of tuples that define possible scalar binary operators

    Args:
      matrix (MatrixDefinition): The matrix to create binary operators for.

    Returns:
      list: List of four-tuples containing return value, operands and operator
    """

    variants = ['matrix', 'array', 'lattice_matrix', 'lattice_array']
    ops = []

    for variant in variants:
        typename = getattr(matrix, "{}_name".format(variant))

        for op in "+*-":
            for scalar in ["Real", "Complex"]:
                ops.extend([
                    (typename, op, scalar, typename),
                    (typename, op, typename, scalar)])
    return ops


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
    operator_includes = []

    for matrix in matrices:
        fnames = [_camel2underscores(getattr(matrix, "{}_name".format(variant)))
                  for variant in variants]
        includes = dict([("{}_include".format(variant), fname)
                         for variant, fname in zip(variants, fnames)])
        scalar_binary_ops.extend(make_scalar_binary_ops(matrix))
        for variant, fname in zip(variants, fnames):
            name = getattr(matrix, "{}_name".format(variant))
            template = env.get_template("core/{}.pxd".format(variant))
            operator_includes.append((fname, name))
            path = os.path.join(output_path, fname + ".pxd")
            print("Writing pyQCD/templates/core/{}.pxd to {}"
                  .format(variant, path))
            with open(path, 'w') as f:
                f.write(template.render(precision=precision, matrixdef=matrix,
                                        includes=includes))

    lattice_binary_ops = []

    for matrix_lhs, matrix_rhs in product(matrices, matrices):
        lattice_binary_ops.extend(
            make_lattice_binary_ops(matrices, matrix_lhs, matrix_rhs))

    path = os.path.join(output_path, "types.hpp")
    types_template = env.get_template("core/types.hpp")
    print("Writing pyQCD/templates/core/types.hpp to {}".format(path))
    with open(path, 'w') as f:
        f.write(types_template.render(matrices=matrices, precision=precision))

    path = os.path.join(output_path, "operators.pxd")
    cython_operator_template = env.get_template("core/operators.pxd")
    print("Writing pyQCD/templates/core/operators.pxd to {}".format(path))
    with open(path, 'w') as f:
        f.write(cython_operator_template.render(
            scalar_binary_ops=scalar_binary_ops,
            lattice_binary_ops=lattice_binary_ops,
            includes=operator_includes))


env.filters['to_underscores'] = _camel2underscores