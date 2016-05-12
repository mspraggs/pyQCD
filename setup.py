from itertools import product
import subprocess
import sys

from Cython.Build import cythonize
from Cython.Compiler.Errors import CompileError
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand

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


extensions = [Extension("pyQCD.core.core", ["pyQCD/core/core.pyx",
                                            "pyQCD/core/comms.cpp",
                                            "pyQCD/core/layout.cpp",
                                            "pyQCD/core/detail/layout_helpers.cpp"],
                        language="c++", undef_macros=["NDEBUG"],
                        include_dirs=["./pyQCD", "/usr/include/eigen3"],
                        extra_compile_args=["-std=c++11", "-DUSE_MPI"])]

# Do not rebuild on change of extension module in the case where we're
# regenerating the code (in case of errors)
if "codegen" in sys.argv:
    ext_modules = []
else:
    ext_modules = cythonize(extensions)


# Test command for interfacing with py.test
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments for py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


# Overridden build command to integrate with SCons
class SConsBuild(build_ext):
    def run(self):
        p = subprocess.Popen(["./scons.py", "-C", "pyQCD"])
        p.communicate()

setup(
    name='pyQCD',
    version='',
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
