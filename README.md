# pyQCD

**Master**|[![Travis CI Build Status](https://travis-ci.org/mspraggs/pyQCD.svg?branch=master)](https://travis-ci.org/mspraggs/pyQCD)
:---:|:---:
**Development**|[![Travis CI Build Status](https://travis-ci.org/mspraggs/pyQCD.svg?branch=devel)](https://travis-ci.org/mspraggs/pyQCD)

pyQCD is an extensible Python package for building and testing lattice
field theory measurements on desktop and workstation computers.

## Features

- Native implementations of lattice QCD algorithms using C++11, exposed to
  Python using Cython with full NumPy interoperability.
- Complete implementation of Wilson's formulation of lattice QCD by way of both
  fermionic and gluonic actions.
- Rectangle-improved gauge actions (Symanzik and Iwasaki).
- Conjugate gradient sparse matrix solver (with and without even-odd
  preconditioning).
- Support for multiple lattice layouts (currently lexicographic and even-odd
  site orderings).
- Functionality to reconfigure package for different precisions and numbers of
  colours.
- Pseudo-heatbath gauge field update algorithm.
- Shared-memory parallelism using OpenMP.

## Installation

*Building and testing has only been conducted only on Debian-based Linux OSes, 
so YMMV. Please report any installation bugs using GitHub's issue tracker and
I'll do my best to help you out. Build variables (flags, include search paths
etc.) may be found in* pyQCD/utils/build/\_\_init\_\_.py.

First you'll need a couple of things:
- C++ compiler supporting the C++11 standard and GCC build flags (e.g. GCC,
  clang, MinGW, etc.);
- Eigen 3 linear algebra library, either by installing via your package manager
  or by downloading the headers from [here](http://eigen.tuxfamily.org).
  
Once you have these requirements, you're ready to install. Create a virtual
environment if that's your thing, then run this command in the root of the pyQCD
package tree:

    pip install .
    
If you don't have pip installed then the following command should definitely
work:

    python setup.py install
    
If you want to run the unit tests, you'll need CMake version 2.8 or above. Run:

    mkdir build && cd build
    cmake .. && make run_tests
    pyQCD/tests/run_tests
    
Basic tests are also available for the Python code using py.test. Simply run
`py.test` in the package's source directory once you've installed the package.
    
## Usage

Python code and C++ extensions are currently spread across four modules:
- pyQCD.core: basic types, including colour vectors and matrices, their lattice
  equivalents and layout classes;
- pyQCD.gauge: gauge actions and gauge field observables;
- pyQCD.fermions: fermion actions;
- pyQCD.algorithms: gauge field update algorithms and sparse matrix solvers.

Create a gauge field on a lattice with spatial extent L = 8 and temporal extent
T = 16 and update it 100 times using Wilson's gauge action and the heatbath
algorithm:

```python
from pyQCD.algorithms import Heatbath
from pyQCD.gauge import WilsonGaugeAction, average_plaquette, cold_start

# Create a gauge field with all matrices set to the identity, laid out
# lexicographically
gauge_field = cold_start([16, 8, 8, 8])
# Create a Wilson gauge action instance with beta = 5.5
action = WilsonGaugeAction(5.5, gauge_field.layout)
# Create a heatbath update instance
heatbath_updater = Heatbath(gauge_field.layout, action)

# Update the gauge field 100 times
heatbath_updater.update(gauge_field, 100)
# Show link at (t, x, y, z) = (0, 0, 0, 0) and mu = 0
print(gauge_field[0, 0, 0, 0, 0])
```
    
Now use the resulting gauge field to solve the Dirac equation using Wilson's
fermion action and even-odd preconditioned conjugate gradient

```python
from pyQCD.algorithms import conjugate_gradient_eoprec
from pyQCD.core import EvenOddLayout, LatticeColourVector
from pyQCD.fermions import WilsonFermionAction

# Gauge field must be even-odd partitioned for even-odd solver to work
lexico_layout = gauge_field.layout
even_odd_layout = EvenOddLayout(lexico_layout.shape)
gauge_field.change_layout(even_odd_layout)

# Create the Wilson action with dimensionless mass of 0.4 and periodic
# boundary conditions
fermion_action = WilsonFermionAction(0.4, gauge_field, [0] * 4)

# Create a point source
rhs = LatticeColourVector(lexico_layout, 4)
rhs[0, 0, 0, 0, 0, 0] = 1.0
rhs.change_layout(even_odd_layout)

# Solve Dirac equation
sol, num_iterations, err = conjugate_gradient_eoprec(
    fermion_action, rhs, 1000, 1e-10)
print("Finsihed solving after {} iterations. Solution has error = {}"
      .format(num_iterations, err))

# Change layout of sol back to lexicographic to enable numpy-functionality
sol.change_layout(lexico_layout)
# Print solution at (t, x, y, z) = (0, 0, 0, 0)
print(sol[0, 0, 0, 0])
```

## Reconfiguring for Different Theories

By default pyQCD supports conventional lattice QCD with three colours. The
source code can be reconfigured to use a different number of colours, like so:

    python setup.py codegen --num-colours=[number of colours]
    
This will regenerate the Cython code that interfaces the underlying C++ code
with Python. You'll then need to compile and install the extension modules again
by rerunning `pip install .` or `python setup.py install`.

The `codegen` command currently also accepts `--precision` and
`--representation` flags. Note that the package currently only supports
QCD in the fundamental representation. Supported precision options are "double"
(the default) and "float". 
