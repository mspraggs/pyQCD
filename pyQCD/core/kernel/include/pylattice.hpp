#ifndef PYLATTICE_HPP
#define PYLATTICE_HPP

#define BOOST_PYTHON_MAX_ARITY 20

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/raw_function.hpp>
#include "lattice.hpp"
#include "gil.hpp"
#include "pyutils.hpp"
#include "linear_operators.hpp"
#include <string>

namespace py = boost::python;

class pyLattice: public Lattice
{
  friend struct lattice_pickle_suite;
public:
  pyLattice(const int spatialExtent = 4,
	    const int temporalExtent = 8,
	    const double beta = 5.5,
	    const double ut = 1.0,
	    const double us = 1.0,
	    const double chi = 1.0,
	    const int action = 0,
	    const int nCorrelations = 50,
	    const int updateMethod = 0,
	    const int parallelFlag = 1,
	    const int chunkSize = 4,
	    const int randSeed = -1);
  pyLattice(const pyLattice& pylattice);
  ~pyLattice();

  double computePlaquetteP(const py::list site2,const int mu, const int nu);
  double computeRectangleP(const py::list site2,const int mu, const int nu);
  double computeTwistedRectangleP(const py::list site2, const int mu,
				  const int nu);
  double computeWilsonLoopP(const py::list cnr, const int r, const int t,
			    const int dim, const int nSmears = 0,
			    const double smearingParameter = 1.0);
  double computeAverageWilsonLoopP(const int r, const int t,
				   const int nSmears = 0,
				   const double smearingParameter = 1.0);
  py::list computeWilsonPropagatorP(const double mass, const py::list site,
				    const int nSmears,
				    const double smearingParameter,
				    const int sourceSmearingType,
				    const int nSourceSmears,
				    const double sourceSmearingParameter,
				    const int sinkSmearingType,
				    const int nSinkSmears,
				    const double sinkSmearingParameter,
				    const int solverMethod,
				    const py::list boundaryConditions,
				    const int precondition,
				    const int maxIterations,
				    const double tolerance,
				    const int verbosity);
  py::list applyWilsonDiracOperator(py::list psi, const double mass,
				    py::list boundaryConditions,
				    const int precondition);
  py::list applyJacobiSmearingOperator(py::list psi, const int numSmears,
				       const double smearingParameter,
				       py::list boundaryConditions);
  void runThreads(const int nUpdates, const int remainder);
  py::list getLinkP(const py::list link);
  void setLinkP(const py::list link, const py::list matrix);
  py::list getRandSu3(const int index) const;
};

#endif
