#include <Eigen/Dense>
#include <complex>
#include <boost/python.hpp>
using namespace Eigen;

class Lattice
{
public:
  Lattice(const int n = 8,
	  const double beta = 5.5,
	  const int Ncor = 50,
	  const int Ncf = 1000,
	  const double eps = 0.24);

  ~Lattice();
  double P(const int site[4], const in mu, const int nu);
  double Pav();
  double Si(const int link[5]);
  Matrix3cd randomSU3();
  void update();

private:
  int n, Ncor, Ncf;
  double beta, eps;
  Matrix3cd***** links;
  
};

Lattice::Lattice(const int n = 8,
		 const double beta = 5.5,
		 const int Ncor = 50,
		 const int Ncf = 1000,
		 const double eps = 0.24)
{
  /*Default constructor. Assigns function arguments to member variables
   and initializes links.*/
}

Lattice::~Lattice()
{
  /*Destructor*/
}

BOOST_PYTHON_MODULE(lattice)
{
  PyEval_InitThread();
  class_<Lattice>("Lattice", init<optional<int,double,int,int,double> >())
}
