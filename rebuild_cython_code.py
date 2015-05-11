import shutil
import sys

import os

from pyQCD.utils.codegen import create_matrix_definition, generate_cython_types


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


def main(num_colours, precision, representation, dest=None):
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
        raise ValueError("Unknown representation")

    src = os.path.join(os.path.dirname(__file__), "pyQCD")
    if dest:
        shutil.copytree(src, dest, ignore=_filter_lib)
    else:
        dest = src

    generate_cython_types(os.path.join(dest, "core"), precision,
                          matrix_definitions)


if __name__ == "__main__":

    try:
        num_colours = int(sys.argv[1])
        precision = sys.argv[2]
        representation = sys.argv[3]
    except IndexError:
        print("Usage: {} <num_colours> <precision> <representation> "
              "[output_path]".format(sys.argv[0]))
    else:
        try:
            output_path = sys.argv[4]
        except IndexError:
            output_path = None
        main(num_colours, precision, representation, output_path)