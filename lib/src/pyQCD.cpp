#include <pylattice.hpp>
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <iostream>

namespace py = boost::python;

// Class that inherits Lattice and provides python wrapper functions

struct lattice_pickle_suite : py::pickle_suite
{
  static py::tuple getinitargs(const pyLattice& pylattice)
  {
    return py::make_tuple(pylattice.nEdgePoints,
			  pylattice.beta_,
			  pylattice.u0_,
			  pylattice.action_,
			  pylattice.nCorrelations,
			  pylattice.rho_,
			  pylattice.epsilon_,
			  pylattice.updateMethod_,
			  pylattice.parallelFlag_);
  }
  
  static py::tuple getstate(const pyLattice& pylattice)
  {
    // Convert the links vector to a list for python compatability
    py::list links;
    for (int i = 0; i < pylattice.nEdgePoints; i++) {
      py::list temp1;
      for (int j = 0; j < pylattice.nEdgePoints; j++) {
	py::list temp2;
	for (int k = 0; k < pylattice.nEdgePoints; k++) {
	  py::list temp3;
	  for (int l = 0; l < pylattice.nEdgePoints; l++) {
	    py::list temp4;
	    for (int m = 0; m < 4; m++) {
	      temp4.append(pylattice.getLinkP(i, j, k, l, m));
	    }
	    temp3.append(temp4);
	  }
	  temp2.append(temp3);
	}
	temp1.append(temp2);
      }
      links.append(temp1);
    }
    
    // Same for the randSU3s
    py::list randSu3s;
    int index = 0;
    for (int i = 0; i < 20; i++) {
      py::list tempList;
      for (int j = 0; j < 20; j++) {
	tempList.append(pylattice.getRandSu3(index));
	index++;
      }
      randSu3s.append(tempList);
    }
    return py::make_tuple(links, randSu3s);
  }
  
  static void setstate(pyLattice& pylattice, py::tuple state)
  {
    if (len(state) != 2) {
      PyErr_SetObject(PyExc_ValueError,
        ("expected 2-item tuple in call to __setstate__; got %s"
	 % state).ptr());
      py::throw_error_already_set();
    }
    
    GaugeField links;
    Sub4Field randSu3s;
    py::list linkStates = py::extract<py::list>(state[0]);
    py::list randSu3States = py::extract<py::list>(state[1]);

    // Convert the compound list of links back to a vector...
    for (int i = 0; i < pylattice.nEdgePoints; i++) {
      SubField temp1;
      for (int j = 0; j < pylattice.nEdgePoints; j++) {
	Sub2Field temp2;
	for (int k = 0; k < pylattice.nEdgePoints; k++) {
	  Sub3Field temp3;
	  for (int l = 0; l < pylattice.nEdgePoints; l++) {
	    Sub4Field temp4;
	    for (int m = 0; m < 4; m++) {
	      Matrix3cd tempMatrix;
	      for (int n = 0; n < 3; n++) {
		for (int o = 0; o < 3; o++) {
		  tempMatrix(n, o) =
		    py::extract<complex<double> >
		    (linkStates[i][j][k][l][m][n][o]);
		}
	      }
	      temp4.push_back(tempMatrix);
	    }
	    temp3.push_back(temp4);
	  }
	  temp2.push_back(temp3);
	}
	temp1.push_back(temp2);
      }
      links.push_back(temp1);
    }
    // And the same for the random SU3 matrices.
    int index = 0;
    for (int i = 0; i < 20; i++) {
      for (int j = 0; j < 20; j++) {
	Matrix3cd tempMatrix;
	for (int k = 0; k < 3; k++) {
	  for (int l = 0; l < 3; l++) {
	    tempMatrix(k, l) =
	      py::extract<complex<double> >(randSu3States[i][j][k][l]);
	  }
	}
	randSu3s.push_back(tempMatrix);
      }
    }
    
    pylattice.links_ = links;
    pylattice.randSu3s_ = randSu3s;
  }
};



double computeAverageWilsonLoopP(py::tuple args, py::dict kwargs)
{
  pyLattice& self = py::extract<pyLattice&>(args[0]);

  py::list keys = kwargs.keys();

  int r = py::extract<int>(kwargs["r"]);
  int t = py::extract<int>(kwargs["t"]);
  int nSmears = py::extract<int>(kwargs["n_smears"]);

  return self.computeAverageWilsonLoopP(r, t, nSmears);
}


  
// Boost python wrapping of the class
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(pyLatticeWOverload,
				       computeWilsonLoopP, 4, 5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(pyLatticeWavOverload, 
				       computeAverageWilsonLoopP, 2, 3)

BOOST_PYTHON_MODULE(pyQCD)
{
  py::class_<pyLattice>("Lattice",
			py::init<py::optional<int, double, double, int, int,
					      double, double, int, int> >())
    .def(py::init<pyLattice&>())
    .def("link", &pyLattice::getLinkP)
    .def("update", &pyLattice::update)
    .def("schwarz_update", &pyLattice::schwarzUpdate)
    .def("next_config", &pyLattice::getNextConfig)
    .def("thermalize", &pyLattice::thermalize)
    .def("plaquette", &pyLattice::computePlaquetteP)
    .def("rectangle", &pyLattice::computeRectangleP)
    .def("twist_rect", &pyLattice::computeTwistedRectangleP)
    .def("wilson_loop", &pyLattice::computeWilsonLoopP,
	 pyLatticeWOverload(py::args("corner", "r", "t", "dimension",
				     "nSmears"),
			    "Calculate Wilson loop"))
    .def("av_plaquette", &pyLattice::computeAveragePlaquette)
    .def("av_rectangle", &pyLattice::computeAverageRectangle)
    .def("av_wilson_loop", py::raw_function(&computeAverageWilsonLoopP, 3))
    .def("av_link", &pyLattice::computeMeanLink)
    .def("print", &pyLattice::print)
    .def("get_rand_su3", &pyLattice::getRandSu3)
    .def_pickle(lattice_pickle_suite())
    .def_readonly("n_cor", &pyLattice::nCorrelations)
    .def_readonly("n_points", &pyLattice::nEdgePoints);
}
