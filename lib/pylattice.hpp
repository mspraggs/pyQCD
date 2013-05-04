#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include "lattice.hpp"

namespace py = boost::python;

class pyLattice: public Lattice
{
  friend struct lattice_pickle_suite;
public:
  pyLattice(const int nEdgePoints = 8,
	    const double beta = 5.5,
	    const int nCorrelations = 50,
	    const int nConfigurations = 1000,
	    const double epsilon = 0.24,
	    const double a = 0.25,
	    const double rho = 0.3,
	    const double u0 = 1,
	    const int action = 0);
  pyLattice(const pyLattice& pylattice);
  ~pyLattice();

  double computePlaquetteP(const py::list site2,const int mu, const int nu);
  double computeRectangleP(const py::list site2,const int mu, const int nu);
  double computeTwistedRectangleP(const py::list site2, const int mu,
				   const int nu);
  double computeWilsonLoopP(const py::list cnr, const int r, const int t,
			     const int dim, const int nSmears = 0);
  double computeAverageWilsonLoopP(const int r, const int t,
				    const int nSmears = 0);
  void runThreads(const int chunkSize, const int nUpdates,
		  const int remainder);
  py::list getLinkP(const int n0, const int n1, const int n2, const int n3,
		     const int dimension) const;
  py::list getRandSu3(const int index) const;
};
