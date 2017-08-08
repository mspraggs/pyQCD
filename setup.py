from functools import partial
from itertools import product
import sys

from Cython.Build import cythonize
from setuptools import Extension, setup, find_packages

from pyQCD.utils.build import PyTest, generate_include_paths
from pyQCD.utils.codegen import CodeGen

# Include/library search hints
file_search_paths = ["/usr/include", "/usr/lib",
                     "/usr/local/include", "/usr/local/lib",
                     "/usr/local", "/usr",
                     "/opt", "/"]

# Generate wildcarded data file include paths
data_dirs = ["pyQCD", "pyQCD/templates"]
include_subdirs = ["core",
                   "core/detail",
                   "fermion_actions",
                   "gauge_actions",
                   "linear_operators",
                   "utils"]
suffixes = [".hpp", ".cpp", ".pxd", ".pyx"]

data_paths = ["{}/{}/*{}".format(dr, subdir, suffix)
              for dr, subdir, suffix in product(data_dirs, include_subdirs,
                                                suffixes)]

# Setup and build extensions

header_search_files = ["signature_of_eigen3_matrix_library"]
include_dirs = ["./pyQCD"]
include_dirs.extend(generate_include_paths(file_search_paths,
                                           header_search_files))

make_extension = partial(Extension, language="c++", undef_macros=["NDEBUG"],
                         include_dirs=include_dirs,
                         extra_compile_args=["-std=c++11", "-O3"],
                         extra_link_args=[])

extension_sources = {
    "pyQCD.core.core":
        ["pyQCD/core/core.pyx", "pyQCD/core/layout.cpp",
         "pyQCD/utils/random.cpp"],
    "pyQCD.fermions.fermions":
        ["pyQCD/fermions/fermions.pyx", "pyQCD/utils/matrices.cpp"],
    "pyQCD.gauge.gauge":
        ["pyQCD/gauge/gauge.pyx"],
    "pyQCD.algorithms.algorithms":
        ["pyQCD/algorithms/algorithms.pyx", "pyQCD/core/layout.cpp",
         "pyQCD/utils/random.cpp"]
}

extensions = [make_extension(module, sources)
              for module, sources in extension_sources.items()]


# Do not rebuild on change of extension module in the case where we're
# regenerating the code (in case of errors)
ext_modules = [] if "codegen" in sys.argv else cythonize(extensions)


with open("README.md") as f:
    long_description = f.read()

setup(
    name='pyQCD',
    version='0.0.0',
    packages=find_packages(exclude=["*test*"]),
    ext_modules=ext_modules,
    url='http://github.com/mspraggs/pyqcd/',
    author='Matt Spraggs',
    author_email='matthew.spraggs@gmail.com',
    cmdclass={'codegen': CodeGen, 'test': PyTest},
    description='pyQCD provides a Python library for running lattice field '
                'theory simulations on desktop and workstation computers.',
    long_description=long_description,
    tests_require=['pytest'],
    package_dir={'': '.'},
    package_data={'pyQCD': data_paths},
    classifiers=[
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        ],
)
