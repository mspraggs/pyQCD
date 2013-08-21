pyQCD
=====
pyQCD provides a Python package for running coarse lattice QCD simulations on desktop and workstation computers.

***Please note that this software is not yet mature, so backwards compatability is not yet guaranteed for newer
versions. If something's broken, have a look at the function reference below, or check the python source code.***

Installation
------------
To build and install the module, the following packages are required:

* boost::python and boost::random, both version 1.49.0 or greater;
* Eigen C++ matrix library, version 3.1.3 or greater;
* numpy version 1.7 or greater;
* scipy version 0.12 or greater;
* OpenMP (version 3), required for parallel updates, but not essential.

Once these are installed, the package can be built using cmake. On Unix-like OSes, this is straightforward:

> cmake .
> make lattice
> make

The package can then be installed:

> sudo make install

I haven't investigated how to do this on Windows, but if anyone has any success with this I'd be interested to
know.

It's possible that one or more of the required packages is not in a standard location, so cmake won't be able
to find these files automatically. In this case, the following flags can be used to point cmake to the relevant
paths:

> -DEIGEN3_INCLUDE_DIR=...
> -DBoost_INCLUDE_DIR=...
> -DBoost_LIBRARY_DIR=...
> -DPYTHON_LIBRARY=...

Quick Start
-----------

The library contains two modules: one to run lattice simulations and one to process the results. Both modules
accept an input xml that specifies the settings for the simulation or processing. Examples of these may be found
in the examples directory, with each file containing a description of what the file encodes. A list of the
default values for the xml input files may be found in the simulation_default.xml file. These values are
automatically inserted when they are missing in the main xml file.

To run a very basic simulation, without measurements, from the project root directory, do

> from pyQCD.bin import simulate
> simulate.main("examples/basic.xml")

Postprocessing can be executed in a similar way:

> form pyQCD.bin import postprocess
> postprocess.main("examples/plaquette_autocorrelation.xml")