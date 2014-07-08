pyQCD Boost.Python Kernel
=========================
Everything under this directory is basically code for a Boost.Python extension
module for that handles the computationally intensive parts of a lattice
simulation. The point of this guide is to summarise the main design philosophy
of this external module so as to provide a starting point for those wishing to
develop this module further.

To those that do develop this module: please make every effort to document your
changes clearly and concisely, and update this document where necessary. This'll
make it easier for others to understand how this all works and hopefully
expedite the development process.

General Points
--------------
The code should be well laid out and conform to best coding practices, for
example those outlined in Scott Meyers' "Effective C++". Full advantage should
be taken of the latest C++ standard features. Use should be made of the Boost
libraries where it significantly improves code readability. At the same time,
the number of dependencies shouldn't be excessive: if you want to add a
dependency just to reduce five lines of code to one, think again.

Dependencies
------------
We make good use of the Boost libraries generally, especially the Boost.Python
library. Eigen 3 is used for linear algebra, as it's fast and vectorized. Here's
a complete dependency list (Debian-like package names in brackets):

* CMake 2.8.8 or greater (cmake)
* Python development headers and library, version 2.6 or greater (python-dev or python3-dev)
* Eigen 3.1.3 or greater (libeigen3-dev)
* Boost.Python 1.46.0 or greater (libboost-python-dev)
* Boost.Random 1.46.0 or greater (libboost-random-dev)
* Boost.Test 1.46.0 or greater (libboost-test-dev)

Optionally:

* OpenMP 3.0 or greater (ships by default on supported compilers)

Building The Module
-------------------
The module can be built in much the same way described in the readme in the root
of this repository. On linux systems:

    cmake .
    make

This will create the binary kernels.so, which is the Python module. In addition,
static and dynamic libraries are built to facilitate developing in C++ alone.
Several benchmarking binaries are created in the benchmarks directory. A series
of tests will also be built in the tests directory.

Code Locations
--------------
Code is divided into several directories that are fairly self-descriptive, but
we outline their contents for completeness:

* benchmarks - source files to generate binaries for benchmarking.
* include - header files
* src - source files
* tests - unit tests

The include and lib directories share a common directory structure:

* base - fundamental types used throughout the source tree
* fermion_actions - pretty self-explanatory
* gauge_actions - also pretty self-explanatory
* linear_operators - basic linear operators from which dirac operators can be built, as well as smearing operators
* utils - utility functions used througout the codebase

The files in src and include should mirror each other, as the latter contains
the headers for the former.