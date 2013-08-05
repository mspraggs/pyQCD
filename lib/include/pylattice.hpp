#ifndef PYLATTICE_HPP
#define PYLATTICE_HPP

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/raw_function.hpp>
#include "lattice.hpp"
#include "gil.hpp"
#include "pyQCD_utils.hpp"
#include <string>

namespace py = boost::python;

class pyLattice: public Lattice
{
  friend struct lattice_pickle_suite;
public:
  pyLattice(const int nEdgePoints = 8,
	    const double beta = 5.5,
	    const double u0 = 1.0,
	    const int action = 0,
	    const int nCorrelations = 50,
	    const double rho = 0.3,
	    const double epsilon = 0.24,
	    const int updateMethod = 0,
	    const int parallelFlag = 1);
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
  py::list computePropagatorP(const double mass, const py::list site,
			      const double spacing, const int solverMethod,
			      const int nSmears);
  void runThreads(const int nUpdates, const int remainder);
  py::list getLinkP(const py::list link);
  void setLinkP(const py::list link, const py::list matrix);
  py::list getRandSu3(const int index) const;
};

#endif
