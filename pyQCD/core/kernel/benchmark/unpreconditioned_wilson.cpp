#include <iostream>

#include <boost/timer/timer.hpp>

#include <lattice.hpp>
#include <linear_operators/unpreconditioned_wilson.hpp>

const int nIterations = 10000;

int main(int argc, char** argv)
{
  Lattice lattice(4, 8, 5.5, 1.0, 1.0, 1.0, 0, 10, 0, 1, 4, -1);

  vector<complex<double> > boundaryConditions(4, complex<double>(1.0, 0.0));
  UnpreconditionedWilson linop(0.4, boundaryConditions, &lattice);

  VectorXcd psi(12 * 4 * 4 * 4 * 8);
  psi(0) = 1.0;

  std::cout << "Performing " << nIterations << " matrix-vector products."
	    << std::endl;

  boost::timer::auto_cpu_timer t;
  for (int i = 0; i < nIterations; ++i)
    VectorXcd eta = linop.apply(psi);

  return 0;
}
