#include <Eigen/Dense>
#include <Eigen/QR>
#include <complex>
#include <boost/python.hpp>
#include <vector>
#include <cstdlib>
#include <iostream>

using namespace Eigen;
using namespace boost::python;
using namespace std;

namespace lattice
{
  int mod(int n, const int d)
  {
    while(n < 0) {
      n += d;
    }
    return n%d;
  }
}

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
  vector<Matrix3cd> randSU3s;
  
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

  this->links = vector< vector< vector< vector< vector<Matrix3cd> > > > > (n,D);

  for(int i = 0; i < 50; i++) {
    Matrix3cd randSU3 = this->randomSU3();
    this->randSU3s.push_back(randSU3);
    this->randSU3s.push_back(randSU3.adjoint());
  }
}

Lattice::~Lattice()
{
  /*Destructor*/  
}

double Lattice::P(const int site[4],const int mu, const int nu)
{
  /*Calculate the plaquette operator at the given site, on plaquette
    specified by directions mu and nu.*/
  int mu_vec[4] = {0,0,0,0};
  mu_vec[mu] = 1;
  int nu_vec[4] = {0,0,0,0};
  nu_vec[nu] = 1;
  int site2[4] = {0,0,0,0};
  int os1[4] = {0,0,0,0};
  int os2[4] = {0,0,0,0};

  for(int i = 0; i < this->n; i++) {
    site2[i] = lattice::mod(site[i], this->n);
    os1[i] = lattice::mod(site[i] + mu_vec[i],this->n);
    os2[i] = lattice::mod(site[i] + nu_vec[i],this->n);
  }

  Matrix3cd product = Matrix3cd::Identity();
  product *= this->links[site2[0]][site2[1]][site2[2]][site2[3]][mu];
  product *= this->links[os1[0]][os1[1]][os1[2]][os1[3]][nu];
  product *= this->links[os2[0]][os2[1]][os2[2]][os2[3]][mu].adjoint();
  product *= this->links[site2[0]][site2[1]][site2[2]][site2[3]][nu].adjoint();

  return 1./3 * product.trace().real();
}

double Lattice::Si(const int link[5])
{
  /*Calculate the contribution to the action from the given link*/
  int planes[3];
  double Psum = 0;

  int j = 0;
  for(int i = 0; i < 4; i++) {
    if(link[4] != i) {
      planes[j] = i;
      j++;
    }
  }

  for(int i = 0; i < 3; i++) {
    int site[4] = {link[0],link[1],link[2],link[3]};
    Psum += this->P(site,link[4],planes[i]);
    site[planes[i]] -= 1;
    Psum += this->P(site,link[4],planes[i]);
  }

  return -this->beta * Psum;
}

Matrix3cd Lattice::randomSU3()
{
  /*Generate a random SU3 matrix, weighted by eps*/

  Matrix3cd A = Matrix3cd::Random();
  Matrix3cd B = Matrix3cd::Identity() + this->eps * A;
  
  ColPivHouseholderQR<Matrix3cd> decomp(B);
  Matrix3cd Q = decomp.householderQ();

  return Q / pow(Q.determinant(),1./3);
}

void Lattice::update()
{
  /*Iterate through the lattice and update the links using Metropolis
    algorithm*/
  for(int i = 0; i < this->n; i++) {
    for(int j = 0; j < this->n; j++) {
      for(int k = 0; k < this->n; k++) {
	for(int l = 0; l < this->n; l++) {
	  for(int m = 0; m < 4; m++) {
	    int link[5] = {i,j,k,l,m};
	    double Si_old = this->Si(link);
	    Matrix3cd linki_old = this->links[i][j][k][l][m];
	    Matrix3cd randSU3 = this->randSU3s[rand() % this->randSU3s.size()];
	    this->links[i][j][k][l][m] *= randSU3;
	    double dS = this->Si(link) - Si_old;
	    
	    if(dS > 0 && exp(-dS) < double(rand()) / double(RAND_MAX)) {
	      this->links[i][j][k][l][m] = linki_old;
	    }
	  }
	}
      }
    }
  }
}

double Lattice::Pav()
{
  /*Calculate average plaquette operator value*/
  int nus[6] = {0,0,0,1,1,2};
  int mus[6] = {1,2,3,2,3,3};
  double Ptot = 0;
  for(int i = 0; i < this->n; i++) {
    for(int j = 0; j < this->n; j++) {
      for(int k = 0; k < this->n; k++) {
	for(int l = 0; l < this->n; l++) {
	  for(int m = 0; m < 6; m++) {
	    int site[4] = {i,j,k,l};
	    Ptot += this->P(site,mus[m],nus[m]);
	  }
	}
      }
    }
  }
  return Ptot / (pow(this->n,4) * 6);
}

BOOST_PYTHON_MODULE(lattice)
{
  class_<Lattice>("Lattice", init<optional<int,double,int,int,double> >())
    .def("update",&Lattice::update)
    .def("Pav",&Lattice::Pav);
}
