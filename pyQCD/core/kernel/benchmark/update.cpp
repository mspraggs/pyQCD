#include <iostream>

#include <boost/timer/timer.hpp>

#include <lattice.hpp>

int main(int argc, char**argv)
{
  int nUpdates = 1000;

  if (argc > 1)
    nUpdates = atoi(argv[1]);
  
  Lattice lattice(8, 8, 5.5, 1.0, 1.0, 1.0, 0, 10, 0, 1, 4, -1);

  std::cout << "Performing " << nUpdates << " heatbath updates." << std::endl;

  boost::timer::cpu_timer timer;
  for (int i = 0; i < nUpdates; ++i) {
    lattice.update();
  }

  boost::timer::cpu_times const elapsedTimes(timer.elapsed());
  boost::timer::nanosecond_type const elapsed(elapsedTimes.system
					      + elapsedTimes.user);
  boost::timer::nanosecond_type const walltime(elapsedTimes.wall);

  std::cout << "Total CPU time = " << elapsed / 1.0e9 << " s" << endl;
  std::cout << "CPU time per update = " << elapsed / 1.0e9 / nUpdates
	    << " s" << endl;
  std::cout << "Walltime = " << walltime / 1.0e9 << " s" << endl;
  std::cout << "Walltime per update = " << walltime / 1.0e9 / nUpdates
	    << " s" << endl;

  return 0;
}
