from itertools import product
import sys

from Cython.Build import cythonize
from Cython.Compiler.Errors import CompileError
from setuptools import Extension, setup, find_packages

from pyQCD.utils.codegen import CodeGen

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


with open("README.md") as f:
    long_description = f.read()


extensions = [Extension("pyQCD.core.core", ["pyQCD/core/core.pyx"],
                        language="c++",
                        include_dirs=["./pyQCD", "/usr/include/eigen3"],
                        extra_compile_args=["-std=c++11"])]

# Do not rebuild on change of extension module in the case where we're
# regenerating the code (in case of errors)
if "codegen" in sys.argv:
    ext_modules = []
else:
    ext_modules = cythonize(extensions)


setup(
    name='pyQCD',
    version='',
    packages=find_packages(exclude=["*test*"]),
    ext_modules=ext_modules,
    url='http://github.com/mspraggs/pyqcd/',
    author='Matt Spraggs',
    author_email='matthew.spraggs@gmail.com',
    cmdclass={'codegen': CodeGen},
    description='pyQCD provides a Python library for running lattice field '
                'theory simulations on desktop and workstation computers.',
    long_description=long_description,
    package_dir={'': '.'},
    package_data={'pyQCD': data_paths},
    classifiers = [
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