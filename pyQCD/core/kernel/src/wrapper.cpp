#define BOOST_PYTHON_MAX_ARITY 20

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/args.hpp>
#include <boost/python/docstring_options.hpp>
#include <iostream>
#include <pylattice.hpp>

namespace py = boost::python;

// Little function for generating list required below

py::list listArg()
{
  py::list out;
  for (int i = 0; i < 4; ++i) 
    out.append(0);
  return out;
}

py::list defaultBoundaryConditions()
{
  py::list out;
  out.append(-1);
  for (int i = 0; i < 3; ++i) 
    out.append(1);
  return out;
}


 
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

  py::class_<pyLattice>("Lattice", py::init<int, int, double, double, double,
					    double, int, int, int, int, int, int>
			((py::arg("L")=4, py::arg("T")=8, py::arg("beta")=5.5,
			  py::arg("ut")=1.0, py::arg("us")=1.0,
			  py::arg("chi")=1.0,
			  py::arg("action")=pyQCD::wilsonPlaquette,
			  py::arg("Ncor")=10,
			  py::arg("update_method")=pyQCD::heatbath,
			  py::arg("parallel_flag")=1, py::arg("block_size")=4,
			  py::arg("rand_seed")=-1)))
    .def(py::init<pyLattice&>())
    .def("get_link", &pyLattice::getLinkP, (py::arg("link")))
    .def("set_link", &pyLattice::setLinkP, (py::arg("link"), py::arg("matrix")))
    .def("update", &pyLattice::update)
    .def("next_config", &pyLattice::getNextConfig)
    .def("thermalize", &pyLattice::thermalize, (py::arg("num_updates")))
    .def("get_plaquette", &pyLattice::computePlaquetteP,
	 (py::arg("site"), py::arg("dim1"), py::arg("dim2")))
    .def("get_rectangle", &pyLattice::computeRectangleP,
	 (py::arg("site"), py::arg("dim1"), py::arg("dim2")))
    .def("get_twist_rect", &pyLattice::computeTwistedRectangleP,
	 (py::arg("site"), py::arg("dim1"), py::arg("dim2")))
    .def("get_wilson_loop", &pyLattice::computeWilsonLoopP,
	 (py::arg("corner"), py::arg("r"), py::arg("t"), py::arg("dim"),
	  py::arg("n_smears") = 0, py::arg("smearing_param") = 1.0 ))
    .def("get_av_plaquette", &pyLattice::computeAveragePlaquette)
    .def("get_av_rectangle", &pyLattice::computeAverageRectangle)
    .def("get_av_wilson_loop", &pyLattice::computeAverageWilsonLoopP,
	 (py::arg("r"), py::arg("t"), py::arg("n_smears") = 0,
	  py::arg("smearing_param") = 1.0))
    .def("get_wilson_propagator", &pyLattice::computeWilsonPropagatorP,
	 (py::arg("mass"), py::arg("site") = listArg(),
	  py::arg("n_link_smears") = 0, py::arg("link_param") = 1.0,
	  py::arg("src_smear_type") = pyQCD::jacobi, py::arg("n_src_smears") = 0,
	  py::arg("src_param") = 1.0, py::arg("sink_smear_type") = pyQCD::jacobi,
	  py::arg("n_sink_smears") = 0, py::arg("sink_param") = 1.0,
	  py::arg("solver_method") = pyQCD::cg,
	  py::arg("boundary_conditions") = defaultBoundaryConditions(),
	  py::arg("precondition") = 0, py::arg("max_iterations") = 1000,
	  py::arg("tolerance") = 1, py::arg("verbosity") = 0))
    .def("get_hamberwu_propagator", &pyLattice::computeHamberWuPropagatorP,
	 (py::arg("mass"), py::arg("site") = listArg(),
	  py::arg("n_link_smears") = 0, py::arg("link_param") = 1.0,
	  py::arg("src_smear_type") = pyQCD::jacobi, py::arg("n_src_smears") = 0,
	  py::arg("src_param") = 1.0, py::arg("sink_smear_type") = pyQCD::jacobi,
	  py::arg("n_sink_smears") = 0, py::arg("sink_param") = 1.0,
	  py::arg("solver_method") = pyQCD::cg,
	  py::arg("boundary_conditions") = defaultBoundaryConditions(),
	  py::arg("precondition") = 0, py::arg("max_iterations") = 1000,
	  py::arg("tolerance") = 1, py::arg("verbosity") = 0))
    .def("apply_wilson_dirac", &pyLattice::applyWilsonDiracOperator,
	 (py::arg("psi"), py::arg("mass"),
	  py::arg("boundary_conditions") = defaultBoundaryConditions(),
	  py::arg("precondition") = 0))
    .def("apply_hamberwu_dirac", &pyLattice::applyHamberWuDiracOperator,
	 (py::arg("psi"), py::arg("mass"),
	  py::arg("boundary_conditions") = defaultBoundaryConditions(),
	  py::arg("precondition") = 0))
    .def("apply_jacobi_smearing", &pyLattice::applyJacobiSmearingOperator,
	 (py::arg("psi"), py::arg("num_smears"), py::arg("smearing_parameter"),
	  py::arg("boundary_conditions") = defaultBoundaryConditions()))
    .def("get_av_link", &pyLattice::computeMeanLink)
    .def_readonly("num_cor", &pyLattice::nCorrelations)
    .def_readonly("L", &pyLattice::spatialExtent)
    .def_readonly("T", &pyLattice::temporalExtent);
}
