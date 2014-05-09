pyQCD
=====
pyQCD provides a Python package for running small-scale lattice simulations on desktop and workstation
computers, as well as tools for analyzing the results.

Features
--------
pyQCD brings an object-orientated approach to lattice simulations and analysis. Simulation features:

* Base Lattice object written in C++ and compiled using Boost.Python provides a flexible and fast API.
* Full parallelisation of CPU-intensive code using OpenMP.
* Capable of using CUDA enabled GPUs to accelerate inversion of Dirac matrices.
* Wilson, Hamber-Wu and Naik 4D Wilson-like fermionic actions.
* Fermionic measurement API provides extensibility through functions for applying Dirac operators and inverting on single quark sources.
* Propagator generation functions for all 4D fermionic actions, with support for Jacobi smearing of sinks and sources.
* Flexible domain wall action that can utilise any existing 4D kernel accepting a single mass parameter.
* Complete integration with numpy ndarray type, facilitating the use of numpy and scipy linear algebra packages.
* Support for multiple gauge actions, including Wilson and Symanzik rectangle-improved.
* Support for stout smearing gauge fields prior to measurements.
* Wilson loop computation capability.
* Convenience functions for saving and loading gauge fields as numpy ndarrays and numpy binaries.

Analysis features:

* Fully parallelised bootstrap and jackknife resampling functions.
* Import functions to load data produced by CHROMA and UKhadron.
* General correlator fitting function to fit an arbitrary function to a given correlator.
* Specialised correlator fitting function to fit the ground state of a given correlator.
* Effective mass computation function.
* Lattice spacing computation using the Sommer scale and static quark pair potential.

***Please note that this software is very much still in alpha stage and is not yet mature, so backwards
compatability is not yet guaranteed for newer versions. If something's broken, have a look at the function
reference below, or check the python source code.***

Installation for Analysis Alone
-------------------------------
To get set up with the analysis components of the package only (i.e. without any tools to generate gauge
configurations, compute propagators and the like) the following packages are required:

* Python setuptools, tested with version 0.6.24, though may work with older versions;
* numpy version 1.7 or greater;
* scipy version 0.12 or greater.

To test the module and build the docs, you'll need the following:

* py.test for Python testing;
* Python Sphinx along with the Napoleon extension.

Installation should then be straightforward (administrator priviledges may be needed):

    python setup.py install

Installation for Simulation and Analysis
----------------------------------------
To build and use the simulation components of the package, you will need the following packages in addition
to those above:

* CMake 2.6 or greater
* boost::python, boost::random, boost::timer and boost::system, all version 1.49.0 or greater;
* Eigen C++ matrix library, version 3.1.3 or greater;
* OpenMP (version 3), required for parallel updates, but not essential.

pyQCD is capable of using CUDA-capable GPUs to accelerate the inversion of Dirac matrices to generate
propagators. Note that preconditioning, tadpole improvement and anisotropic lattices are currently not
implemented in the cuda linear operator kernels. If you have a CUDA enabled GPU, and you want to use its
capabilities, then you'll also need the following packages:

* CUDA, version 4.2 or greater;
* CUSP Sparse Matrix Library, available [here](http://cusplibrary.github.io/), using commit 6cde5ee.

Once these are installed, the package can be built using cmake. On Unix-like OSes, this is straightforward:

    cmake .
    make

To see substantial performance gains, you can tell the compiler to use AVX and turn off the Eigen bounds
checking code by running cmake as follows before compiling:

    cmake . -DCMAKE_CXX_FLAGS="-mavx -DNDEBUG"

The "-mavx" flag will likely vary between compilers. See your compilers documentation for details.

If you want to use CUDA, then you'll need to specify the path to CUSP when you run cmake. For example,
if you clone the CUSP library into /home/user/cusplibrary, then you'll need to run:

    cmake . -DCUSP_INCLUDE_DIR=/home/user/cusplibrary
    make

If you don't want to use CUDA, then you can either omit the CUSP library include path, in which case cmake
will fall back to Eigen's sparse matrix inverters, or use the flag -DUSE_CUDA=0 when running cmake.

Then proceed as above:

    python setup.py install

I haven't investigated how to do this on Windows, but if anyone has any success with this I'd be interested to
know.

It's possible that one or more of the required packages is not in a standard location, so cmake won't be able
to find these files automatically. In this case, the following flags can be used to point cmake to the relevant
paths:

    -DEIGEN3_INCLUDE_DIR=...
    -DBoost_INCLUDE_DIR=...
    -DBoost_LIBRARY_DIR=...
    -DPYTHON_LIBRARY=...

Running the Tests
-----------------
Once everything's built and installed, it's a good idea to test it. To test the code, just run

    py.test pyQCD

in the project root directory or

    py.test

in the module root directory.

Building the Docs
-----------------
To build the documentation on linux, enter the docs directory and run

    make html

On Windows:

    make.bat html

This will build html documentation in the _build/html dubdirectory. At the moment this provides a brief function
reference. In the future this will hopefully be more detailed.

Benchmarking the Package
------------------------
To benchmark the package on your machine, run the benchmark.py script in the project root directory. It is
possible to to provide a string command line argument to filter certain benchmarking functions.

Quick Start
-----------
Several examples are included in the examples/ directory in the repository root. I'd encourage newcomers
to look there for ideas on how to use pyQCD. I'd also strongly encourage the use of IPython when exploring
the package (http://ipython.org/).
