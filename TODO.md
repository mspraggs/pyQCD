pyQCD Feature and Bug Todo List
-------------------------------

- Add constructor argument flag to determine whether lattice is initialized hot or cold (i.e. random unitary matrices or unit matrices).
- Add correlation function calculation to the postprocess script.
- Optimize average wilson loop calculation to remove repeated calculation of Wilson lines.
- Add numpy support to pylattice.cpp using the Boost.Numpy module on github.
- Add anti-periodic in time boundary conditions functionality.
- Allow specification of number of thermalization updates in xml file.
