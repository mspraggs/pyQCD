pyQCD Boost.Python Kernel
=========================
The pyQCD.Lattice class contains several functions that perform computationally intensive operations,
such as updating the lattice and inverting Dirac matrices. To make these operations faster, the code
is written in C++ and wrapped using Boost.Python. This guide is written as a brief guide for those
who want to develop this code further.

Building The Module
-------------------
Building the module is done in exactly the same way as when you're in the root of the pyQCD
repository:

    cmake .
    make

This will create the binary lattice.so, which is the Python module containing the Lattice class.
Alongside this, static and dynamic libraries are built to facilitate developing in C++ alone.
Several benchmarking binaries are created in the benchmark directory. There should also be a
testing binary, though this is somewhat deprecated in favour of testing the code via Python.

Code Locations
--------------
Code is divided into three directories: include, src and benchmark, containing (surprise, surprise)
header files, source files and code for benchmarking various functions. Contents of the header
directory:

* gil.hpp - class to aid in scoping the release of Python's global interpreter lock.
* lattice.hpp - contains the core C++ implementation of the Lattice object.
* linear_operators.hpp - umbrella include file for the contents of the linear_operators directory.
* pylattice.hpp - wrapper class that inherits the class in lattice.hpp and converts Python types to C++ types.
* pyutils.hpp - functions for converting python types, particularly lists, to C++ types.
* random.hpp - contains a thread-safe wrapper to the boost random number generator.
* solvers.hpp - contains iterative linear solver functions.
* utils.hpp - contains utility functions and constants, such as the gamma matrices.

The linear_operators sub-directory contains Dirac and smearing operators. Of particular note are
the files linear_operator.hpp and hopping_term.hpp. The former contains the base linear operator
class from which all others are derived. The latter is a generalised central difference hopping
operator, allowing one to specify an arbitrary number of hops, with an arbitrary spin structure
for each of the eight directions in which the operator acts.

The contents of the src directory shadow those in the include directory. Since there are many
lines of source code in the Lattice class, they are split into several files based on their
function:

* lattice_ctors.cpp - class constructors and desctructors
* lattice_ferm_meas.cpp - fermionic measurements, including propagator computation routines and single source inversion functions.
* lattice_gauge_acts.cpp - gauge actions, including Wilson and rectangle-improved actions.
* lattice_glue_meas.cpp - gluonic measurements, including Wilson loop and plaquette functions.
* lattice_update.cpp - algorithms to update the gauge field.
* lattice_utils.cpp - utililty functions relating to an instance of the lattice object.

Conventions
-----------
Eigen 3 is used as the principle package for the linear algebra required in the computations.

If a set of coordinates are specified by an integer array or std::vector, the zeroth element of
these objects will correspond to the time coordinate. Here we're breaking with the lattice convention
of labelling time as the fourth dimension, but given that computers start counting at zero, it makes
more sense, at least to my mind, to label the axis using the Minkowski convention.

Currently all the data is arranged lexicographically, with the frequency of variation of the indices
on objects decreasing in the following order: colour, spin, axis, z-coordinate, y-coordinate,
x-coordinate, t-coordinate. For example, the variable holding the gauge links in the Lattice class is
a vector of Eigen Matrix3cd objects (i.e. 3x3 complex double valued matrices). The position of a given
link matrix denoted by a dimension mu and coordinates x, y, z and t would then be at the following
position in this vector:

    index = mu + 4 * z + 4 * L * y + 4 * L^2 * x + 4 * L^3 * t

Similarly, spinor objects are specified by vectors in the LinearOperator class and its child classes.
A given complex value in this array is indexed in the following way (alpha and a are the spin and
colour indices, respectively):

    index = c + 3 * alpha + 12 * L * y + 12 * L^2 * x + 12 * L^3 * t