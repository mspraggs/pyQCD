pyQCD Feature and Bug Todo List
-------------------------------

- Add constructor argument flag to determine whether lattice is initialized hot or cold (i.e. random unitary matrices or unit matrices).
- Add correlation function calculation to the postprocess script.
- Adjust propagator arguments so that some have a default (e.g. spacing could have default of 1).
- Optimize average wilson loop calculation to remove repeated calculation of Wilson lines.
- Add flexibility in spatial and time extents (i.e. L and T instead of N)
- Add numpy support to pylattice.cpp using the Boost.Numpy module on github.
- Add anti-periodic in time boundary conditions functionality.
- Use string arguments for settings like solver method, update method, etc., rather than integers.
- Refactor postprocessing functions into statistics and measurements files.
