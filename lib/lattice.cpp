#include <Eigen/Dense>
#include <complex>
#include <boost/python.hpp>
#include <vector>

using namespace Eigen;
using namespace boost::python;
using namespace std;

class Lattice
{
public:
  Lattice(const int n = 8,
	  const double beta = 5.5,
	  const int Ncor = 50,
	  const int Ncf = 1000,
	  const double eps = 0.24);

  ~Lattice();
  double P(const int site[4], const int mu, const int nu);
  double Pav();
  double Si(const int link[5]);
  Matrix3cd randomSU3();
  void update();

private:
  int n, Ncor, Ncf;
  double beta, eps;
  vector< vector< vector< vector< vector<Matrix3cd> > > > > links;
  
};

Lattice::Lattice(const int n, const double beta, const int Ncor, const int Ncf, const double eps)
{
  /*Default constructor. Assigns function arguments to member variables
   and initializes links.*/
  this->n = n;
  this->beta = beta;
  this->Ncor = Ncor;
  this->Ncf = Ncf;
  this->eps = eps;

  vector<Matrix3cd> A (4,Matrix3cd::Identity());
  vector< vector<Matrix3cd> > B (n,A);
  vector< vector< vector<Matrix3cd> > > C (n,B);
  vector< vector< vector< vector<Matrix3cd> > > > D (n,C);

  this->links = vector< vector< vector< vector< vector<Matrix3cd> > > > > (m,D);
}

Lattice::~Lattice()
{
  /*Destructor*/
  
}

BOOST_PYTHON_MODULE(lattice)
{
  class_<Lattice>("Lattice", init<optional<int,double,int,int,double> >());
}
