from itertools import product
import sys

from setuptools import setup, find_packages

from pyQCD.utils.build import PyTest, generate_extensions, generate_libraries
from pyQCD.utils.build.build_shared_clib import BuildSharedLib
from pyQCD.utils.build.build_ext import BuildExt
from pyQCD.utils.codegen import CodeGen

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

libraries = generate_libraries(sys.argv)
ext_modules = generate_extensions(sys.argv)

with open("README.md") as f:
    long_description = f.read()

setup(
    name='pyQCD',
    version='0.0.0',
    packages=find_packages(exclude=["*test*"]),
    ext_modules=ext_modules,
    libraries=libraries,
    url='http://github.com/mspraggs/pyqcd/',
    author='Matt Spraggs',
    author_email='matthew.spraggs@gmail.com',
    cmdclass={'codegen': CodeGen, 'test': PyTest, "build_ext": BuildExt,
              'build_clib': BuildSharedLib},
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
