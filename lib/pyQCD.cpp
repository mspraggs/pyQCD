#include "lattice.cpp"

/*The python wrapper for the C++ class*/

struct my_pickle_suite : py::pickle_suite
{
  static py::tuple getinitargs(const Lattice& L)
  {
    return py::make_tuple(L.n,L.beta,L.Ncor,L.Ncf,L.eps,L.a,L.smear_eps,L.u0,L.action);
  }
  
  static py::tuple getstate(const Lattice& L)
  {
    //Convert the links vector to a list for python compatability
    py::list links;
    for(int i = 0; i < L.n; i++) {
      py::list temp1;
      for(int j = 0; j < L.n; j++) {
	py::list temp2;
	for(int k = 0; k < L.n; k++) {
	  py::list temp3;
	  for(int l = 0; l < L.n; l++) {
	    py::list temp4;
	    for(int m = 0; m < 4; m++) {
	      temp4.append(L.getLink(i,j,k,l,m));
	    }
	    temp3.append(temp4);
	  }
	  temp2.append(temp3);
	}
	temp1.append(temp2);
      }
      links.append(temp1);
    }
    
    //Same for the randSU3s
    py::list randSU3s;
    int index = 0;
    for(int i = 0; i < 20; i++) {
      py::list temp_list;
      for(int j = 0; j < 20; j++) {
	temp_list.append(L.getRandSU3(index));
	index++;
      }
      randSU3s.append(temp_list);
    }
    return py::make_tuple(links,randSU3s);
  }
  
  static void setstate(Lattice& L, py::tuple state)
  {
    if(len(state) != 2) {
      PyErr_SetObject(PyExc_ValueError,
		      ("expected 2-item tuple in call to __setstate__; got %s"
		       % state).ptr());
      py::throw_error_already_set();
    }
    
    vector< vector< vector< vector< vector<Matrix3cd, aligned_allocator<Matrix3cd> > > > > > links;
    vector<Matrix3cd, aligned_allocator<Matrix3cd> > randSU3s;
    py::list link_states = py::extract<py::list>(state[0]);
    py::list randSU3_states = py::extract<py::list>(state[1]);

    //Convert the compound list of links back to a vector...
    for(int i = 0; i < L.n; i++) {
      vector< vector< vector< vector<Matrix3cd, aligned_allocator<Matrix3cd> > > > > temp1;
      for(int j = 0; j < L.n; j++) {
	vector< vector< vector<Matrix3cd, aligned_allocator<Matrix3cd> > > > temp2;
	for(int k = 0; k < L.n; k++) {
	  vector< vector<Matrix3cd, aligned_allocator<Matrix3cd> > > temp3;
	  for(int l = 0; l < L.n; l++) {
	    vector<Matrix3cd, aligned_allocator<Matrix3cd> > temp4;
	    for(int m = 0; m < 4; m++) {
	      Matrix3cd temp_mat;
	      for(int n = 0; n < 3; n++) {
		for(int o = 0; o < 3; o++) {
		  temp_mat(n,o) = py::extract<complex<double> >(link_states[i][j][k][l][m][n][o]);
		}
	      }
	      temp4.push_back(temp_mat);
	    }
	    temp3.push_back(temp4);
	  }
	  temp2.push_back(temp3);
	}
	temp1.push_back(temp2);
      }
      links.push_back(temp1);
    }
    //And the same for the random SU3 matrices.
    int index = 0;
    for(int i = 0; i < 20; i++) {
      for(int j = 0; j < 20; j++) {
	Matrix3cd temp_mat;
	for(int k = 0; k < 3; k++) {
	  for(int l = 0; l < 3; l++) {
	    temp_mat(k,l) = py::extract<complex<double> >(randSU3_states[i][j][k][l]);
	  }
	}
	randSU3s.push_back(temp_mat);
      }
    }
    
    L.links = links;
    L.randSU3s = randSU3s;
  }
};
  
//Boost python wrapping of the class
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(LatticeWOverload,W_p,4,5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(LatticeWavOverload,Wav,2,3)

BOOST_PYTHON_MODULE(pyQCD)
{
  py::class_<Lattice>("Lattice", py::init<py::optional<int,double,int,int,double,double,double,double,int> >())
    .def(py::init<Lattice&>())
    .def_pickle(my_pickle_suite())
    .def("update",&Lattice::update)
    .def("nextConfig",&Lattice::nextConfig)
    .def("thermalize",&Lattice::thermalize)
    .def("init_u0",&Lattice::init_u0)
    .def("P",&Lattice::P_p)
    .def("Pav",&Lattice::Pav)
    .def("R",&Lattice::R_p)
    .def("Rav",&Lattice::Rav)
    .def("T",&Lattice::T_p)
    .def("W",&Lattice::W_p,LatticeWOverload(py::args("cnr","r","t","dim","n_smears"), "Calculate Wilson loop"))
    .def("Wav",&Lattice::Wav,LatticeWavOverload(py::args("r","t","n_smears"), "Calculate average Wilson loop"))
    .def("printL",&Lattice::printL)
    .def("getLink",&Lattice::getLink)
    .def("getRandSU3",&Lattice::getRandSU3)
    .def_readonly("Ncor",&Lattice::Ncor)
    .def_readonly("Ncf",&Lattice::Ncf)
    .def_readonly("n",&Lattice::n);
}
