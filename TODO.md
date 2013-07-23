pyQCD Feature and Bug Todo List
-------------------------------

- Add constructor argument flag to determine whether lattice is
initialized hot or cold (i.e. random unitary matrices or unit matrices).
- Add function to allow gauge field to be loaded into a Lattice object
from a python list (and perhaps a file in interfaces to extract the field
from a numpy array or something).
- Implementation of smeared sources in the propagator inversion.
- Add propagator support to the run script.
- Add correlation function calculation to the postprocess script.
- Change zero-momentum function to project propagator onto arbitrary
momentum.
- Adjust propagator arguments so that some have a default (e.g. spacing
could have default of 1).
- Refactor code to use utility functions where possible in pyQCD_utils.
- Make sure all code is properly commented and meets language conventions.
