pyQCD
=====
pyQCD provides a Python library for running coarse lattice QCD simulations on desktop and workstation computers.

Installation
------------
Please note I haven't yet invested the time in proper deployment scripts for this project, so at this stage compiling
this library for use on your platform may prove a challenge.

pyQCD requires the following for compilation to work:

* Boost::Python (tested with versions 1.44 and 1.49, so anything greater than 1.44 should work)
* Eigen C++ matrix library (version 3 or greater)
* OpenMP (version 3 or greater)

Once these are installed, enter the lib directory. A Makefile is provided for use with make. Depending on your system,
you may find you need to change the locations of the include files and libraries for boost::python, Python and Eigen
(please use the variables specified within "else" segment of the Makefile).

If you are building the library on the IRIDIS cluster machine, you will need to download the Eigen library from the web
(no compilation should be necessary) and adjust the include path for this library. You will also need to load the boost
module and use gcc version 4.6.1 to get OpenMP support. To get the Makefile to recognised that you're on IRIDIS, do
> export IRIDIS=true

Once the include paths have been set up, just run make and with any luck you should end up with a pyQCD shared library
in the lib directory.

Getting Started
---------------
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
