#include <Eigen/Dense>
#include <complex>
#include <boost/python.hpp>
using namespace Eigen;
using namespace boost::python;

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
  Matrix3cd***** links;
  
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

  //Initialize the array of matrices
  this->links = new Matrix3cd**** [n];

  for(int i = 0; i < n; i++) {
    this->links[i] = new Matrix3cd*** [n];
    
    for(int j = 0; j < n; j++) {
      this->links[i][j] = new Matrix3cd** [n];
      
      for(int k = 0; k < n; k++) {
	this->links[i][j][k] = new Matrix3cd* [n];

	for(int l = 0; l < n; l++) {
	  this->links[i][j][k][l] = new Matrix3cd[4];

	  for(int m = 0; m < 4; m++) {
	    this->links[i][j][k][l][m] = Matrix3cd::Identity();
	  }	  
	}
      }
    }
  }
}

Lattice::~Lattice()
{
  /*Destructor*/
}

BOOST_PYTHON_MODULE(lattice)
{
  class_<Lattice>("Lattice", init<optional<int,double,int,int,double> >());
}
