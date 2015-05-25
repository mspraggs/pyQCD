"""This module contains various utility functions. Specifically:

- Functions to generate Cython code a specific number of colours, representation
etc.
"""

from __future__ import absolute_import, print_function

import os
from collections import namedtuple
from itertools import product
import shutil
from string import lowercase

from jinja2 import Environment, PackageLoader
import setuptools


# Create the jinja2 template environment.
env = Environment(loader=PackageLoader('pyQCD', 'templates'),
                  trim_blocks=True, lstrip_blocks=True)

MatrixDefinition = namedtuple("MatrixDefinition",
                              ["num_rows", "num_cols", "matrix_name",
                               "array_name", "lattice_matrix_name",
                               "lattice_array_name"])

variants = ['matrix', 'array', 'lattice_matrix', 'lattice_array']


def _filter_lib(src, names):
    """Filters out C++ and Cython files from list of names"""

    out = []
    for name in names:
        if name in ["build", "cmake", "CMakeFiles"]:
            out.append(name)
        elif os.path.isdir(os.path.join(src, name)):
            pass
        elif name[-4:] not in ['.hpp', '.cpp', '.pxd', '.pyx']:
            out.append(name)
    return out


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
      tuple: Resulting matrix shape, allowed variants and permitted operations

      The first element in the tuple will be the shape of the resulting matrix,
      the second will be a list of three-tuple containing the permitted
      lhs and rhs types and the corresponding return type.
    """

    can_sum = ((matrix_lhs.num_cols == matrix_rhs.num_cols)
               and (matrix_lhs.num_rows == matrix_rhs.num_rows))
    can_mult = matrix_lhs.num_cols == matrix_rhs.num_rows

    if not (can_mult or can_sum):
        return (-1, -1), [], False, False
    else:
        result_shape = (matrix_lhs.num_rows, matrix_rhs.num_cols)

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

    return result_shape, pairs, can_sum, can_mult


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
    ret_shape, variant_triplets, can_sum, can_mult \
        = get_compatible_variants(matrix_lhs, matrix_rhs)
    ret_lookup = dict([((m.num_rows, m.num_cols), m) for m in matrices])
    try:
        matrix_ret = ret_lookup[ret_shape]
    except KeyError:
        return []
    for vartrip in variant_triplets:
        lhs_name, rhs_name, ret_name = tuple([
            getattr(mat, "{}_name".format(var))
            for mat, var in zip([matrix_lhs, matrix_rhs, matrix_ret], vartrip)
        ])
        can_sub = (
            ("lattice" in vartrip[0] if "lattice" in vartrip[1] else True) and
            ("array" in vartrip[0] if "array" in vartrip[1] else True)
            and can_sum
        )
        lhs_name = "{}.{}".format(_camel2underscores(lhs_name), lhs_name)
        rhs_name = "{}.{}".format(_camel2underscores(rhs_name), rhs_name)
        ret_name = "{}.{}".format(_camel2underscores(ret_name), ret_name)
        opcodes = (('*' if can_mult else '') + ('+' if can_sum else '') +
                   ('-' if can_sub else ''))
        for op in opcodes:
            print(ret_name, "=", lhs_name, op, rhs_name)
            ops.append((ret_name, op, lhs_name, rhs_name))

    return ops


def make_scalar_binary_ops(matrix, precision):
    """Create a list of tuples that define possible scalar binary operators

    Args:
      matrix (MatrixDefinition): The matrix to create binary operators for.

    Returns:
      list: List of four-tuples containing return value, operands and operator
    """

    ops = []

    for variant in variants:
        typename = getattr(matrix, "{}_name".format(variant))
        typename = "{}.{}".format(_camel2underscores(typename), typename)

        for scalar in [precision, "complex.Complex"]:
            ops.extend([
                (typename, "*", scalar, typename),
                (typename, "*", typename, scalar),
                (typename, '/', typename, scalar)])
    return ops


def make_cython_ops(matrices, cpp_ops, precision):
    """Convert a list of operator tuples from C++ to Cython description

    This means partitionining the list according the matrix type of the
    operands, in addition to converting the character defining the arithmetic
    operator to the appropriate Python function name.

    Args:
      matrices (list): A list of MatrixDefinition instances defining the
        matrices that the operators should be build for.
      cpp_ops (list): List of four-tuples specifying the arithmetic operators,
        as returned by the make_scalar_binary_ops and make_lattice_binary_ops
        functions.
      precision (str): The fundamental machine type to be used throughout the
        code (e.g. 'double' or 'float').
    """

    func_lookup = {'+': 'add', '-': 'sub',
                   '*': 'mul', '/': 'div'}

    out = dict([((getattr(mat, "{}_name".format(var)), op), [])
                for mat in matrices for var in variants
                for op in func_lookup.values()])
    for ret_type, op, lhs_type, rhs_type in cpp_ops:
        ret_type = ret_type.split('.')[-1]
        lhs_type = lhs_type.split('.')[-1]
        lhs_type = 'float' if lhs_type == precision else lhs_type
        rhs_type = rhs_type.split('.')[-1]
        rhs_type = 'float' if rhs_type == precision else rhs_type
        key = (lhs_type if lhs_type not in ['float', 'Complex'] else rhs_type,
               func_lookup[op])
        out[key].append((ret_type, lhs_type, rhs_type))

    return out


def write_core_template(template_fname, output_fname, output_path,
                        **template_args):
    """Load the specified template from templates/core and render it to core"""

    template = env.get_template("core/{}".format(template_fname))
    path = os.path.join(output_path, output_fname)
    print("Writing pyQCD/templates/core/{} to {}".format(template_fname, path))
    with open(path, 'w') as f:
        f.write(template.render(**template_args))


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

    # List of tuples of allowed binary operations
    scalar_binary_ops = []
    lattice_binary_ops = []
    operator_includes = []

    for matrix in matrices:
        fnames = [_camel2underscores(getattr(matrix, "{}_name".format(variant)))
                  for variant in variants]
        includes = dict([("{}_include".format(variant), fname)
                         for variant, fname in zip(variants, fnames)])
        scalar_binary_ops.extend(make_scalar_binary_ops(matrix, precision))
        for variant, fname in zip(variants, fnames):
            name = getattr(matrix, "{}_name".format(variant))
            operator_includes.append((fname, name))
            write_core_template(variant + ".pxd", fname + ".pxd", output_path,
                                precision=precision, matrixdef=matrix,
                                includes=includes)

    for matrix_lhs, matrix_rhs in product(matrices, matrices):
        lattice_binary_ops.extend(
            make_lattice_binary_ops(matrices, matrix_lhs, matrix_rhs))

    cython_ops = make_cython_ops(
        matrices, scalar_binary_ops + lattice_binary_ops, precision)

    write_core_template("types.hpp", "types.hpp", output_path,
                        matrixdefs=matrices, precision=precision)
    write_core_template("complex.pxd", "complex.pxd", output_path,
                        precision=precision)
    write_core_template("operators.pxd", "operators.pxd", output_path,
                        scalar_binary_ops=scalar_binary_ops,
                        lattice_binary_ops=lattice_binary_ops,
                        includes=operator_includes)
    write_core_template("core.pyx", "core.pyx", output_path,
                        matrixdefs=matrices, operators=cython_ops,
                        precision=precision)


def generate_qcd(num_colours, precision, representation, dest=None):
    """Main script entry point

    Builds MatrixDefinition instances and passes them to generate_cython_types.

    Arguments:
      num_colours (int): Number of colours to use in matrix definitions.
      precision (str): The fundamental floating point type to use in
        calculations, e.g. "single", "double", etc.
      representation (str): The gauge field representation to use. Currently
        only 'fundamental' is supported.
      lib_dest (str): The root path in which to output the cython code.
        Defaults to the lib directory in the project root directory.
    """

    matrix_definitions = []
    if representation == "fundamental":
        matrix_definitions.append(create_matrix_definition(
            num_colours, num_colours, "ColourMatrix",
            lattice_array_name="GaugeField"
        ))
        matrix_definitions.append(create_matrix_definition(
            num_colours, 1, "ColourVector", array_name="Fermion",
            lattice_array_name="FermionField"
        ))
    else:
        raise ValueError("Unknown representation: {}".format(representation))

    src = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                        "../../pyQCD"))
    if dest:
        shutil.copytree(src, dest, ignore=_filter_lib)
    else:
        dest = src

    generate_cython_types(os.path.join(dest, "core"), precision,
                          matrix_definitions)


class CodeGen(setuptools.Command):

    description = "Generate Cython code."
    user_options = [
        ("num-colours=", "c", "Number of colours (defaults to 3)"),
        ("precision=", "p", "Fundamental type for real numbers "
                            "(defaults to double)"),
        ("representation=", "r", "Representation (defaults to fundamental)")]

    def initialize_options(self):
        """Initialize options to their default values"""
        self.num_colours = 3
        self.precision = "double"
        self.representation = "fundamental"

    def finalize_options(self):
        """Finalize options - convert num_colours to int"""
        if isinstance(self.num_colours, str):
            self.num_colours = int(self.num_colours)

    def run(self):
        """Run - pass execution to generate_qcd"""
        generate_qcd(self.num_colours, self.precision, self.representation)


env.filters['to_underscores'] = _camel2underscores
env.globals.update(zip=zip)