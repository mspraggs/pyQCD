#include <Eigen/Dense>
#include <Eigen/QR>
#include <complex>
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <ctime>

using namespace Eigen;
using namespace boost::python;
using namespace std;

namespace lattice
{
  const complex<double> i (0,1);
  const double pi = 3.1415926535897932384626433;
  
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
  Matrix3cd calcPath(const vector<vector<int> > path);
  double P(const int site[4], const int mu, const int nu);
  double Pav();
  double Si(const int link[5]);
  Matrix3cd randomSU3();
  void update();
  void printL();
  Matrix3cd link(const int link[5]);
  list getLink(const int i, const int j, const int k, const int l, const int m);

  int Ncor, Ncf, n;

private:
  double beta, eps;
  vector< vector< vector< vector< vector<Matrix3cd, aligned_allocator<Matrix3cd> > > > > > links;
  vector<Matrix3cd, aligned_allocator<Matrix3cd> > randSU3s;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
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

  srand(time(NULL));
  this->links.resize(this->n);
  for(int i = 0; i < this->n; i++) {
    this->links[i].resize(this->n);
    for(int j = 0; j < this->n; j++) {
      this->links[i][j].resize(this->n);
      for(int k = 0; k < this->n; k++) {
	this->links[i][j][k].resize(this->n);
	for(int l = 0; l < this->n; l++) {
	  this->links[i][j][k][l].resize(4);
	  for(int m = 0; m < 4; m++) {
	    this->links[i][j][k][l][m] = this->randomSU3();
	  }
	}
      }
    }
  }
  
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

Matrix3cd Lattice::link(const int[5] link)
{
  /*Return link specified by index (sanitizes link indices)*/
  int link2[5];
  for(int i = 0; i < 5; i++) {
    link2[i] = lattice::mod(link[i],this->n);
  }
  return this->links[link2[0]][link2[1]][link2[2]][link2[3]][link2[4]];
}

Matrix3cd Lattice::calcPath(const vector<vector<int> > path)
{
  /*Multiplies the matrices together specified by the indices in path*/
  Matrix3cd out = Matrix3cd::Identity();
  
  for(int i = 0; i < path.size() - 1; i++) {
    //Which dimension are we moving in?
    int dim = path[i][4];
    if(path[i+1][dim] - path[i][dim] == 0) {
      // Consecutive points don't match link direction, so throw and error
    }
  }
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

  for(int i = 0; i < 4; i++) {
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

  Matrix3cd A;
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      A(i,j) = double(rand()) / double(RAND_MAX);
      A(i,j) *= exp(2 * lattice::pi * lattice::i * double(rand()) / double(RAND_MAX));
    }
  }
  Matrix3cd B = Matrix3cd::Identity() + lattice::i * this->eps * A;
  HouseholderQR<Matrix3cd> decomp(B);
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
	    
	    if((dS > 0) && (exp(-dS) < double(rand()) / double(RAND_MAX))) {
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

void Lattice::printL()
{
  for(int i = 0; i < this->n; i++) {
    for(int j = 0; j < this->n; j++) {
      for(int k = 0; k < this->n; k++) {
	for(int l = 0; l < this->n; l++) {
	  for(int m = 0; m < 4; m++) {
	    cout << this->links[i][j][k][l][m] << endl;
	  }
	}
      }
    }
  }
}

list Lattice::getLink(const int i, const int j, const int k, const int l, const int m)
{
  list out;
  for(int n = 0; n < 3; n++) {
    list temp;
    for(int o = 0; o < 3; o++) {
      temp.append(this->links[i][j][k][l][m](n,o));
    }
    out.append(temp);
  }
  return out;
}

BOOST_PYTHON_MODULE(lattice)
{
  class_<Lattice>("Lattice", init<optional<int,double,int,int,double> >())
    .def("update",&Lattice::update)
    .def("Pav",&Lattice::Pav)
    .def("printL",&Lattice::printL)
    .def("getLink",&Lattice::getLink)
    .def_readonly("Ncor",&Lattice::Ncor)
    .def_readonly("Ncf",&Lattice::Ncf)
    .def_readonly("n",&Lattice::n);
}
