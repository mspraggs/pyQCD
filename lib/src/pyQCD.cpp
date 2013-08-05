#include <pylattice.hpp>
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/args.hpp>
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
  
  static py::tuple getstate(pyLattice& pylattice)
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
	      py::list tempList;
	      tempList.append(i);
	      tempList.append(j);
	      tempList.append(k);
	      tempList.append(l);
	      tempList.append(m);
	      temp4.append(pylattice.getLinkP(tempList));
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
    GaugeField randSu3s;
    py::list linkStates = py::extract<py::list>(state[0]);
    py::list randSu3States = py::extract<py::list>(state[1]);

    // Convert the compound list of links back to a vector...
    for (int i = 0; i < pylattice.nEdgePoints; i++) {
      for (int j = 0; j < pylattice.nEdgePoints; j++) {
	for (int k = 0; k < pylattice.nEdgePoints; k++) {
	  for (int l = 0; l < pylattice.nEdgePoints; l++) {
	    for (int m = 0; m < 4; m++) {
	      Matrix3cd tempMatrix;
	      for (int n = 0; n < 3; n++) {
		for (int o = 0; o < 3; o++) {
		  tempMatrix(n, o) =
		    py::extract<complex<double> >
		    (linkStates[i][j][k][l][m][n][o]);
		}
	      }
	      links.push_back(tempMatrix);
	    }
	  }
	}
      }
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


  
// Boost python wrapping of the class
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(pyLatticeWOverload,
				       computeWilsonLoopP, 4, 5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(pyLatticeWavOverload, 
				       computeAverageWilsonLoopP, 2, 3)

BOOST_PYTHON_MODULE(pyQCD)
{
  py::class_<pyLattice>("Lattice",
			py::init<int, double, double, int, int, double,
				 double, int, int>
			((py::arg("n")=8, py::arg("beta")=5.5,
			  py::arg("u0")=1.0, py::arg("action")=0,
			  py::arg("Ncor")=10, py::arg("rho")=0.3,
			  py::arg("epsilon")=0.24, py::arg("update_method")=0,
			  py::arg("parallel_flag")=1)))
    .def(py::init<pyLattice&>())
    .def("get_link", &pyLattice::getLinkP, (py::arg("link")))
    .def("set_link", &pyLattice::setLinkP, (py::arg("link")))
    .def("update", &pyLattice::update)
    .def("schwarz_update", &pyLattice::schwarzUpdate, (py::arg("n_sweeps")=1))
    .def("next_config", &pyLattice::getNextConfig)
    .def("thermalize", &pyLattice::thermalize)
    .def("plaquette", &pyLattice::computePlaquetteP,
	 (py::arg("site"), py::arg("dim1"), py::arg("dim2")))
    .def("rectangle", &pyLattice::computeRectangleP,
	 (py::arg("site"), py::arg("dim1"), py::arg("dim2")))
    .def("twist_rect", &pyLattice::computeTwistedRectangleP,
	 (py::arg("site"), py::arg("dim1"), py::arg("dim2")))
    .def("wilson_loop", &pyLattice::computeWilsonLoopP,
	 (py::arg("corner"), py::arg("r"), py::arg("t"), py::arg("dim"),
	  py::arg("n_smears") = 0))
    .def("av_plaquette", &pyLattice::computeAveragePlaquette)
    .def("av_rectangle", &pyLattice::computeAverageRectangle)
    .def("av_wilson_loop", &pyLattice::computeAverageWilsonLoopP,
	 (py::arg("r"), py::arg("t"), py::arg("n_smears") = 0))
    .def("propagator", &pyLattice::computePropagatorP,
	 (py::arg("mass"), py::arg("site"), py::arg("spacing"),
	  py::arg("solver_method") = 0, py::arg("n_smears") = 0))
    .def("av_link", &pyLattice::computeMeanLink)
    .def("print", &pyLattice::print)
    .def("get_rand_su3", &pyLattice::getRandSu3,
	 (py::arg("index")))
    .def_pickle(lattice_pickle_suite())
    .def_readonly("n_cor", &pyLattice::nCorrelations)
    .def_readonly("n_points", &pyLattice::nEdgePoints);
}
