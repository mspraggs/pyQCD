"""This module contains the main entry point for the Cython code generator. It
calls various other utilities within the codegen package.
"""

from __future__ import absolute_import, print_function

import os
from collections import namedtuple
from itertools import product
import shutil
from string import ascii_lowercase

from jinja2 import Environment, PackageLoader
import setuptools

from . import typedefs
from .typedefs import LatticeDef


# Create the jinja2 template environment.
env = Environment(loader=PackageLoader('pyQCD', 'templates'),
                  trim_blocks=True, lstrip_blocks=True,
                  extensions=["jinja2.ext.do", "jinja2.ext.with_"])

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
    for char in ascii_lowercase:
        string = string.replace(char.upper(), '_' + char)
    return string.lstrip('_')


def create_type_definitions(num_rows, num_cols, matrix_name):
    """Create container type definitions based on matrix type

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
      list: A named tuple containing the supplied parameters.
    """

    lattice_matrix_name = "Lattice{}".format(matrix_name)

    complex_type = typedefs.TypeDef("Complex", "Complex", "atomics")
    shape = (num_rows, num_cols) if num_cols > 1 else (num_rows,)

    matrix_def = typedefs.MatrixDef(
        matrix_name, matrix_name, _camel2underscores(matrix_name), shape,
        complex_type)

    lattice_matrix_def = typedefs.LatticeDef(
        lattice_matrix_name, lattice_matrix_name,
        _camel2underscores(lattice_matrix_name), matrix_def)

    return [matrix_def, lattice_matrix_def]


def write_template(template_fname, output_fname, output_path, **template_args):
    """Load the specified template from templates/core and render it to core"""

    template = env.get_template(template_fname)
    path = os.path.join(output_path, output_fname)
    print("Writing pyQCD/templates/{} to {}".format(template_fname, path))
    with open(path, 'w') as f:
        f.write(template.render(**template_args))


def generate_core_cython_types(output_path, precision, typedefs, operator_map):
    """Generate Cython source code for matrix, array and lattice types.

    Code is generated from templates in the package template directory using the
    jinja2 templating engine.

    Args:
      output_path (str): The root output directory in which to put the generated
        code.
      precision (str): The fundamental machine type to be used throughout the
        code (e.g. 'double' or 'float').
      typdefs (iterable): An iterable object containing instances of TypeDef.
      operator_map (dict): Dictionary relating arithmetic operator characters
        to lists of Python function names that implement them.
    """

    templates = ["core/qcd_types.hpp", "core/atomics.pxd",
                 "core/core.pyx", "core/core.pxd"]

    for template in templates:
        write_template(template, template, output_path,
                       typedefs=typedefs, precision=precision,
                       operator_map=operator_map)

    return [os.path.join(output_path, template) for template in templates]


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

    operator_map = {"*": ["mul"], "/": ["div", "truediv"],
                    "+": ["add"], "-": ["sub"]}

    type_definitions = []
    if representation == "fundamental":
        type_definitions.extend(create_type_definitions(
            num_colours, num_colours, "ColourMatrix"
        ))
        type_definitions.extend(create_type_definitions(
            num_colours, 1, "ColourVector"
        ))
    else:
        raise ValueError("Unknown representation: {}".format(representation))

    src = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                        "../../../pyQCD"))
    if dest:
        shutil.copytree(src, dest, ignore=_filter_lib)
    else:
        dest = src

    cython_files = []

    cython_files.extend(generate_core_cython_types(
        dest, precision, type_definitions, operator_map))

    write_template("globals.hpp", "globals.hpp", dest,
                   num_colours=num_colours, precision=precision)

    return [path for path in cython_files if path.endswith(".pyx")]


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
