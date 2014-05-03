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
* Convenience functions for saving and loading gauge fields in numpy zip format.

Analysis features:

* Fully-fledged DataSet class, providing parallelised bootstrap and jackknife resampling functions.
* Resampling routines allow for extensibility by accepting user-created measurement functions.
* TwoPoint correlator wrapper class provides core fitting and Propagator contraction routines.
* TwoPoint class fully capable of importing correlator data from Chroma and UKhadron.
* WilsonLoops class facilitates computation of static quark pair potential and lattice spacing via the Sommer scale.

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
* boost::python, boost::random and boost::test, all version 1.49.0 or greater;
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
Once everything's built and installed, it's a good idea to test it. First run the boost test executable:

    make test

Then test the Python implementation:

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

Simulations are performed using the Simulation class. The settings for an individual simulation may be
specified individually. For example:

    import pyQCD
    
    simulation = pyQCD.Simulation(num_configs=100,
                                  measurement_spacing=10,
                                  num_warmup_updates=100,
                                  update_method="heatbath",
                                  run_parallel=True,
                                  rand_seed=-1,
                                  verbosity=2)

    simulation.create_lattice(L=4, T=8, action="wilson", beta=5.5, u0=1.0)
    simulation.run()

The plaquettes for each of the individual configurations are then stored in the simulation object as the member
"plaquettes", so in the above cases simulation.plaquettes contain the sequence of plaquttes values from the set
of generated configurations.

Measurements can be added to the simulation using the add_measurement function as follows:

    .
    .
    .
    simulation.add_measurement(pyQCD.Lattice.get_propagator,
                               pyQCD.Propagator,
			       "propagators.zip",
			       kwargs={"mass": 0.4})
    simulation.run()

Results are saved in zip archives that can be opened using the pyQCD.DataSet object. The results may be loaded
and analyzed in an object-orientated manner, as follows:

    import pyQCD
    
    propagators = pyQCD.DataSet.load("propagators.zip")
    correlators = pyQCD.DataSet(pyQCD.TwoPoint, "correlators.zip")
    
    for i in xrange(propagators.num_data):
        # Specify the propagators that make up the two-point function
	twopoint = pyQCD.TwoPoint(L=4, T=8)
	twopoint.compute_meson_correlator(propagators[i], propagators[i], pyQCD.gamma5, pyQCD.gamma5, "pion")
        correlators.add_datum(twopoint)
    # Compute the energies of the zero-momentum pion and rho_x with a fit range of 2 <= t <= 6
    correlator_mean, correlator_std = correlators.statistics()
    energies = correlators.jackknife(pyQCD.TwoPoint.compute_energy,
                                     args=([2, 6], [1.0, 1.0], correlator_std.get_correlator("pion"), "pion"))
