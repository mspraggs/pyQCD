#include <iostream>

#include <boost/timer/timer.hpp>

#include <lattice.hpp>
#include <linear_operators/naik.hpp>

const int nIterations = 10000;

int main(int argc, char** argv)
{
  int nIterations = 10000;

  if (argc > 1) {
    nIterations = atoi(argv[1]);
  }

  Lattice lattice(4, 8, 5.5, 1.0, 1.0, 1.0, 0, 10, 0, 1, 4, -1);

  vector<complex<double> > boundaryConditions(4, complex<double>(1.0, 0.0));
  Naik linop(0.4, boundaryConditions, &lattice);

  VectorXcd psi = VectorXcd::Zero(12 * 4 * 4 * 4 * 8);
  psi(0) = 1.0;

  std::cout << "Performing " << nIterations << " matrix-vector products."
	    << std::endl;

  boost::timer::cpu_timer timer;
  for (int i = 0; i < nIterations; ++i) {
    VectorXcd eta = linop.apply(psi);
  }

  boost::timer::cpu_times const elapsedTimes(timer.elapsed());
  boost::timer::nanosecond_type const elapsed(elapsedTimes.system
					      + elapsedTimes.user);
  boost::timer::nanosecond_type const walltime(elapsedTimes.wall);

  std::cout << "Total CPU time = " << elapsed / 1.0e9 << " s" << endl;
  std::cout << "CPU time per iteration = " << elapsed / 1.0e9 / nIterations
	    << " s" << endl;
  std::cout << "Walltime = " << walltime / 1.0e9 << " s" << endl;
  std::cout << "Walltime per iteration = " << walltime / 1.0e9 / nIterations
	    << " s" << endl;
  std::cout << "Performance: " << linop.getNumFlops()
	    << " floating point operations; "
	    << (double) linop.getNumFlops() / elapsed * 1000.0
	    << " MFlops / thread" << endl;

  return 0;
}
