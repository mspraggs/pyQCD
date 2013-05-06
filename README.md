pyQCD
=====
pyQCD provides a Python library for running coarse lattice QCD simulations on desktop and workstation computers.

Installation
------------
pyQCD can be built using the cmake build system. The CMakeLists.txt file has only been tested on Ubuntu Linux,
so I can't guarantee that it'll work seamlessly on other platforms (obviously this is something I'd like to change
in the future).

pyQCD requires the following for compilation to work:

* Boost::Python (tested with versions 1.44 and 1.49, so anything greater than 1.44 should work)
* Eigen C++ matrix library (version 3.1.3 or greater)
* OpenMP (version 3 or greater)
* Numpy and scipy Python libraries (not required for compilation, but for the python scripts supplied)

Once you have all these, enter the lib directory and run:

> cmake .

> make

If cmake fails then you'll need to check the generated CMakeCache.txt file to ensure that the Python, Boost and Eigen
library and include paths are set correctly (search for each of these terms within the file).

This should build the shared library in the lib directory.

If you're using this library on the Iridis cluster, you'll need to download a copy of Eigen (no compilation required)
and point cmake in the direction of the Eigen and Python include files. You'll also need to load gcc 4.6.1 for OpenMP
to work correctly and boost 1.44.0. A command like this should generate a Makefile that works:

> cmake . -DEIGEN3_INCLUDE_DIR=/path/to/eigen3 -DPYTHON_INCLUDE_PATH=/local/software/rh53/python/2.6.5/gcc/include/python2.6 -DPYTHON_LIBRARY=/local/software/rh53/python/2.6.5/gcc/lib/libpython2.6.a

Getting Started
---------------
***Note, if you're using Iridis you'll need to load numpy and gcc 4.6.1 for the run.py script to work***

To run a basic simulation, all you need to do is return to the home directory of the project and execute run.py. You
should make a results folder in the project root directory to store the results files. This will run the following set
up:

* 8^4 lattice
* Wilson gauge action
* beta / u0 = 5.5 (u0 here is the tadpole improvement factor for coarse lattices)
* 1000 configuration measurements with 50 lattice updates between configurations
* No link smearing on measurements

At the moment the run script measures the plaquette expectation value of each configuration and the expectation values
of all sizes of temporal-spatial Wilson loops for each configuration. These results will be dumped into a directory
names after the run date and time and the simulation parameters. The postprocessing.py script can be used to analyze
the results from the simulation. Running the script should give you an interactive command prompt, allowing you to
select the appropriate dataset, then allowing you to manipulate the data in several ways.

Taking it Further
-----------------
It's highly unlikely you'll be content to run with the default run script configuration, so here's a summary of the 
command line parameters for the run script:

* --n=X, where X is the given number of sites per side of the lattice (so you'll end up with a X^4 size lattice)
* --beta=X, where X is the QCD beta value
* --u0=X, where X is the tadpole improvement factor
* --action=N, N is an integer specifying the lattice gauge action (0 gives Wilson, 1 is rectangle improved, 2 is twisted rectangle improved)
* --nsmears=N, where N is the number of spatial link smears made during measurements
* --Ncor=N, where N is the number of updates between measurements
* --Ncf=N, where N is the number of configurations generated
* --eps=X, where X is the update tuning factor (leave this at 0.24 to get a 50% update acceptance rate
* --spacing=X, where X is the lattice spacing (at this stage this won't alter how the simulation works)
* --rho=X, where X is the smearing weighting factor (0.3 by default)
* -t or --test calculates how long the simulation will take to run (approximately).
