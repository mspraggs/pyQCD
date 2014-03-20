#include <lattice.hpp>
#include <linear_operators/unpreconditioned_wilson.hpp>

int main(int argc, char** argv)
{
  Lattice lattice(4, 8, 5.5, 1.0, 1.0, 1.0, 0, 10, 0, 1, 4, -1);

  vector<complex<double> > boundaryConditions(4, complex<double>(1.0, 0.0));
  UnpreconditionedWilson linop(0.4, boundaryConditions, &lattice);

  VectorXcd psi(12 * 4 * 4 * 4 * 8);
  psi(0) = 1.0;

  for (int i = 0; i < 100; ++i)
    VectorXcd eta = linop.apply(psi);
}
