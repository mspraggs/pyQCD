#include <pylattice.hpp>
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/args.hpp>
#include <boost/python/docstring_options.hpp>
#include <iostream>

namespace py = boost::python;

// Little function for generating list required below

py::list listArg()
{
  py::list out;
  for (int i = 0; i < 4; ++i) 
    out.append(0);
  return out;
}

// Class that inherits Lattice and provides python wrapper functions

struct lattice_pickle_suite : py::pickle_suite
{
  static py::tuple getinitargs(const pyLattice& pylattice)
  {
    return py::make_tuple(pylattice.spatialExtent,
			  pylattice.temporalExtent,
			  pylattice.beta_,
			  pylattice.u0_,
			  pylattice.action_,
			  pylattice.nCorrelations,
			  pylattice.updateMethod_,
			  pylattice.parallelFlag_);
  }
  
  static py::tuple getstate(pyLattice& pylattice)
  {
    // Convert the links vector to a list for python compatability
    py::list links;
    for (int i = 0; i < pylattice.temporalExtent; i++) {
      py::list temp1;
      for (int j = 0; j < pylattice.spatialExtent; j++) {
	py::list temp2;
	for (int k = 0; k < pylattice.spatialExtent; k++) {
	  py::list temp3;
	  for (int l = 0; l < pylattice.spatialExtent; l++) {
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
    for (int i = 0; i < pylattice.temporalExtent; i++) {
      for (int j = 0; j < pylattice.spatialExtent; j++) {
	for (int k = 0; k < pylattice.spatialExtent; k++) {
	  for (int l = 0; l < pylattice.spatialExtent; l++) {
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

BOOST_PYTHON_MODULE(lattice)
{
  py::docstring_options docopt;
  docopt.enable_all();
  docopt.disable_cpp_signatures();

  py::class_<pyLattice>("Lattice",
			"Constructs a lattice object of spatial extent L and "
			"temporal extent T.\nThe action may take a value of 0, "
			"1 or 2, corresponding to Wilson's\ngauge action, a "
			"rectangle-improved Wilson gauge action and a twisted\n"
			"rectangle-improved Wilson gauge action, respectively. "
			"Ncor corresponds\nto the number of updates performed "
			"by the next_config function. If\nparallel_flag is "
			"equal to 1, then parallel updates are performed,\n"
			"splitting the lattice into blocks of length "
			"block_size. The update\nmethod flag may be set to 0, 1 "
			"or 2, corresponding to heatbath updates,\nmetropolis "
			"updates or metropolis updates without the use of "
			"link\nstaples, respectively. If rand_seed = -1, then "
			"the system time is used \nto seed the generator.\n\n"
			"Constructor arguments and default values:\n\n"
			"L = 4\n\n"
			"T = 8\n\n"
			"beta = 5.5\n\n"
			"u0 = 1.0\n\n"
			"action = 0\n\n"
			"Ncor = 10\n\n"
			"update_method = 0\n\n"
			"parallel_flag = 1\n\n"
			"block_size = 4\n\n"
			"rand_seed = -1",
			py::init<int, int, double, double, int, int, int,
				 int, int, int>
			((py::arg("L")=4, py::arg("T")=8, py::arg("beta")=5.5,
			  py::arg("u0")=1.0, py::arg("action")=0,
			  py::arg("Ncor")=10, py::arg("update_method")=0,
			  py::arg("parallel_flag")=1, py::arg("block_size")=4,
			  py::arg("rand_seed")=-1),
			"Constructs a lattice object of spatial extent L and \n"
			"temporal extent T."))
    .def(py::init<pyLattice&>())
    .def("get_link", &pyLattice::getLinkP, (py::arg("link")),
	 "Returns the link specified by the coordinates in link (of the form\n"
	 "[t, x, y, z, mu]), returning a compound list.")
    .def("set_link", &pyLattice::setLinkP, (py::arg("link"), py::arg("matrix")),
	 "Sets the link specified by a list of the form [t, x, y, z, mu] to\n"
	 "the values specified in matrix.")
    .def("update", &pyLattice::update,
	 "Performs a single linear update on the lattice, using the specified\n"
	 "algorithm")
    .def("schwarz_update", &pyLattice::schwarzUpdate, (py::arg("n_sweeps")=1),
	 "Performs a parallel update on the lattice, splitting it into blocks\n"
	 "and updating the blocks in parallel in the manner of a checkerboard.")
    .def("next_config", &pyLattice::getNextConfig,
	 "Updates the lattice Ncor times to generate the next configuration.")
    .def("thermalize", &pyLattice::thermalize, (py::arg("num_updates")),
	 "Updates the lattice until the internal update counter reaches \n"
	 "num_updates.")
    .def("plaquette", &pyLattice::computePlaquetteP,
	 (py::arg("site"), py::arg("dim1"), py::arg("dim2")),
	 "Calculates the plaquette with corner sited at the specified lattice\n"
	 "site (a list of the form [t, x, y, z]), lying in the plane specified\n"
	 "by dimensions dim1 and dim2.")
    .def("rectangle", &pyLattice::computeRectangleP,
	 (py::arg("site"), py::arg("dim1"), py::arg("dim2")),
	 "Calculates the rectangle with corner sited at the specified lattice\n"
	 "site (a list of the form [t, x, y, z]), lying in the plane specified\n"
	 "by dimensions dim1 and dim2. Here dim1 specifies the long edge of \n"
	 "the rectangle, whilst dim2 specifies the short edge.")
    .def("twist_rect", &pyLattice::computeTwistedRectangleP,
	 (py::arg("site"), py::arg("dim1"), py::arg("dim2")),
	 "Calculates the twisted rectangle with corner sited at the specified \n"
	 "lattice site (a list of the form [t, x, y, z]), lying in the plane \n"
	 "specified by dimensions dim1 and dim2. Here dim1 specifies the long\n"
	 "edge of the rectangle, whilst dim2 specifies the short edge.")
    .def("wilson_loop", &pyLattice::computeWilsonLoopP,
	 (py::arg("corner"), py::arg("r"), py::arg("t"), py::arg("dim"),
	  py::arg("n_smears") = 0, py::arg("smearing_param") = 1.0 ),
	 "Calculates the Wilson loop starting from the lattice site specified\n"
	 "by corner (a list of the form [t, x, y, z]) with spatial component\n"
	 "orientated along dimension dim and size r x t. Stout smearing is\n"
	 "performed by setting the number of smears (n_smears) and the \n"
	 "smearing parameter (smearing_param).")
    .def("av_plaquette", &pyLattice::computeAveragePlaquette,
	 "Computes the plaquette expectation value.")
    .def("av_rectangle", &pyLattice::computeAverageRectangle,
	 "Computes the rectangle expectation value.")
    .def("av_wilson_loop", &pyLattice::computeAverageWilsonLoopP,
	 (py::arg("r"), py::arg("t"), py::arg("n_smears") = 0,
	  py::arg("smearing_param") = 1.0),
	 "Computes the Wilson loop expectation value for all loops of size\n"
	 "r x t. Stout link smearing is performed by specifying the number\n"
	 "of link smears (n_smears) and the smearing parameter (smearing_param)")
    .def("propagator", &pyLattice::computePropagatorP,
	 (py::arg("mass"), py::arg("spacing") = 1.0, py::arg("site") = listArg(),
	  py::arg("n_link_smears") = 0, py::arg("link_param") = 1.0,
	  py::arg("n_src_smears") = 0, py::arg("src_param") = 1.0,
	  py::arg("n_sink_smears") = 0, py::arg("sink_param") = 1.0,
	  py::arg("solver_method") = 0, py::arg("verbosity") = 0),
	 "Extracts the Wilson fermion propagator, as calculated for the \n"
	 "specified mass, lattice spacing and source site, as a flattened list\n"
	 "of compound lists. The list index corresponds to the lattice\n"
	 "coordinates t, x, y and z via the following formula:\n\n"
	 "site_index = z + L * y + L**2 * x + L**3 * t\n\n"
	 "where L is the spatial extent of the lattice.\n\n"
	 "It it possible to apply stout smearing to the lattice gauge field\n"
	 "and Jacobi smearing to the propagator source and sink using the \n"
	 "given function arguments.")
    .def("av_link", &pyLattice::computeMeanLink,
	 "Returns the mean link, that is, the expectation value of the trace\n"
	 "of the lattice link.")
    .def("print", &pyLattice::print)
    .def("get_rand_su3", &pyLattice::getRandSu3,
	 (py::arg("index")))
    .def_pickle(lattice_pickle_suite())
    .def_readonly("n_cor", &pyLattice::nCorrelations)
    .def_readonly("L", &pyLattice::spatialExtent)
    .def_readonly("T", &pyLattice::temporalExtent);
}
