pyQCD Feature and Bug Todo List
-------------------------------

- Add constructor argument flag to determine whether lattice is initialized hot or cold (i.e. random unitary matrices or unit matrices).
- Add function to allow gauge field to be loaded into a Lattice object from a python list (and perhaps a file in interfaces to extract the field from a numpy array or something).
- Implementation of smeared sources in the propagator inversion.
- Add correlation function calculation to the postprocess script.
- Adjust propagator arguments so that some have a default (e.g. spacing could have default of 1).
- Refactor code to use utility functions where possible in pyQCD_utils.
- Make sure all code is properly commented and meets language conventions.
- Optimize average wilson loop calculation to remove repeated calculation of Wilson lines.
- Add flexibility in spatial and time extents (i.e. L and T instead of N)
- Add gauge field smearing to Dirac matrix construction
- Add numpy support to pylattice.cpp using the Boost.Numpy module on github.
- Add anti-periodic in time boundary conditions functionality.
