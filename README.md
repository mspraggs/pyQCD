pyQCD
=====
pyQCD provides a Python library for running coarse lattice QCD simulations on desktop and workstation computers.

Installation
------------
***If you're building this on Iridis, you'll need to clone the Iridis branch of this repo, as this contains
some code modifications to prevent segfaults. As a result of these differences, python keyword arguments are not
supported on Iridis.***

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
* --update-method=N, where N defines the method used to update the gauge configurations. Use 0 for heatbath updates, 1 for efficient Monte Carlo updates and 2 for inefficient Monte Carlo updates. Note that the inefficient Monte Carlo method is the only one that'll work with the twisted rectangle operator, since the link staples cannot be computed for the twisted rectangle operator.
* --parallel-flag=N, where N=1 enables parallel updating, while N=0 disables it.
* -t or --test calculates how long the simulation will take to run (approximately).
* -P or --store-plaquette stores the mean plaquette values for each configuration.
* -W or --store-wloop stores the expectation values of the Wilson loop for each gauge configuration.
* -C or --store-configs stores the gauge configurations from the simulation.

Using the Module
----------------
Writing your own lattice simulations with pyQCD is designed to be straightforward, facilitated by the Lattice object. To
get started, import the module. If you're in the main pyQCD directory, run

> import lib.pyQCD as pyQCD

The lattice has the following constructor:

> pyQCD.Lattice(n, beta, u0, action, Ncor, rho, eps, update_method, parallel_flag)

The arguments are defined in a similar way to above:
* n - the number of points along each edge of the lattice (by default, n=8)
* beta - the beta function value for the simulation (by default, beta=5.5)
* u0 - the tadpole improvement value (by default, u0=1)
* action - the action used in the simulation; 0 gives Wilson action, 1 gives rectangle improved action, 2 gives twisted rectangle improved action (by default, action=0)
* Ncor - the number of configs between measurements (by default, Ncor=10)
* rho - the stout smearing factor (by default, rho=0.3)
* eps - the weighting for the random SU(3) matrix generation algorithm (by default, eps=0.24 to give an approximate 50% rejection rate for the update)
* update_method - the method used for updating the gauge configurations, as defined above (by default, update_method=0, except for the twisted rectangle action, where it can only be 2)
* parallel_flag - designates whether OpenMP should be used (by default, it is, with parallel_flag=1)

This constructor and all the constructors that follow now use keyword arguments for convenience.

The lattice object has the following methods:

> lattice.av_link()

This calculate the average value of 1/3 * Tr[U] (i.e. the trace of a lattice link divided by 3) for the current
configuration.

> lattice.av_plaquette()

This calculates the average value of the plaquette operator.

> lattice.av_rectangle()

This calculates the average value of the rectangle operator.

> lattice.av_wilson_loop(r, t, n_smears = 0)

This calculates the average value of a planar Wilson loop of size r x t (in units of lattice spacing) using n_smears.
n_smears is an optional argument, and is by default 0.

> lattice.get_rand_su3(index)

Gets one of the 200 random SU(3) matrices generated on initialisation and returns it as a compound list

> lattice.link(n0, n1, n2, n3, dim)

Gets the link at site (n0, n1, n2, n3) along the axis specified by dim (0 is time, and 1, 2 and 3 are x, y and
z axes).

> lattice.next_config()

This updates the lattice Ncor times to generate the next configuration for measurement

> lattice.plaquette(site, dim1, dim2)

This calculates the plaquette at the site specified by the list site (e.g. [n_t, n_x, n_y, n_z]), lying the in the plane
specified by the axes dim1 and dim2 (again, 0 is time and 1, 2 and 3 are spatial indices).

> lattice.print()

This was designed to print the lattice, but it doesn't work at the moment

> lattice.rectangle(site, dim1, dim2)

This calculates the rectangle operator in the same way as lattice.plaquette calculates the plaquette value.

> lattice.schwarz_update(block_size = 4, n_sweeps = 1)

This runs a single parallel update, dividing the lattice into blocks, each with width block_size. The blocks are split
into two sets so that they form a sort of 4d checkerboard. The blocks in one set are then updated in parallell, before
the blocks in the second set are updated. n_sweeps determines how many sweeps are performed on each block in one update
step. If the parallel_flag in the constructor is 1, then this function is used in the next_config and thermalize
functions.

> lattice.thermalize()

This updates the lattice 5*Ncor times to bring it into thermal equilibrium.

> lattice.twist_rect(site, dim1, dim2)

Similar the plaquette and rectangle functions, but calculates the twisted rectangle operator

> lattice.update()

Performs a single serial update on the entire lattice.

> lattice.wilson_loop(site, r, t, dim, n_smears = 0)

Calculates the Wilson loop with corner at site (e.g. [n_t, n_x, n_y, n_z]) of size r x t in the spatial dimension
specified by dim. n_smears specifies the number of stout smears, equal to 0 by default.
