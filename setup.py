from setuptools import setup, find_packages
from itertools import product
import os

data_dirs = ["lib", "templates"]
include_subdirs = ["base",
                   "base/detail",
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


# Exclude the lattice.py and simulation.py files if lattice.so
# hasn't been built.
setup(
    name='pyQCD',
    version='',
    packages=find_packages(exclude=["*test*"]),
    url='http://github.com/mspraggs/pyqcd/',
    author='Matt Spraggs',
    author_email='matthew.spraggs@gmail.com',
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