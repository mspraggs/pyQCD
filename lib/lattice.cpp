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
  Matrix3cd calcLine(const int start[4], const int finish[4]);
  double W(const int c1[4], const int c2[4]);
  double W_p(const list cnr1, const list cnr2);
  double P(const int site[4], const int mu, const int nu);
  double P_p(const list site2,const int mu, const int nu);
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
  //Resize the link vector and assign each link a random SU3 matrix
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
  //Generate a set of random SU3 matrices for use in the updates
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

Matrix3cd Lattice::link(const int link[5])
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
    int dim_diff = path[i+1][dim] - path[i][dim];
    if(abs(dim_diff) != 1) {
      // Consecutive points don't match link direction, so throw an error
      cout << "Error! Path contains non-consecutive link variables." << endl;
    }
    else if(dim_diff == -1) {
      /*We're going backwards, so the link must be the adjoint of the link 
	matrix, which we get by using the next site on the lattice.*/
      int link[5] = {path[i+1][0],path[i+1][1],path[i+1][2],path[i+1][3],path[i][4]};
      out *= this->link(link).adjoint();
    }
    else {
      //We're going forwards, so it's just the normal matrix
      int link[5] = {path[i][0],path[i][1],path[i][2],path[i][3],path[i][4]};
      out *= this->link(link);
    }
  }

  return out;
}

Matrix3cd Lattice::calcLine(const int start[4], const int finish[4])
{
  /*Multiplies all gauge links along line from start to finish*/
  //First check that's actually a straight path
  Matrix3cd out = Matrix3cd::Identity();
  int count_dims = 0;
  int dim = 0;
  for(int i = 0; i < 4; i++) {
    if(abs(start[i] - finish[i]) != 0) {
      /*Keep trac of the most recent dimension that differs between start
	and finish, as if we're good to go then we'll need this when
	defining the path.*/
      dim = i;
      count_dims++;
    }
  }

  if(count_dims != 1) {
    cout << "Error! Start and end points do not form a straight line." << endl;
  }
  else {
    //If the two points are on the same line, we're good to go
    vector<vector<int> >  line; //Stores the path

    //Now need to know if we're going backwards or forwards
    if(start[dim] > finish[dim]) {
      for(int i = start[dim]; i >= finish[dim]; i--) {
	//Create the link vector to append, and initialize it's elements
	vector<int> link(5);
	link.assign(start,start+4);
	//Update the index that's parallel to the line with the current
	//location
	link[dim] = i;
	//The direction is going to be equal to the direction of the line
	link[4] = dim;
	//Push it onto the line
	line.push_back(link);
      }
      out = this->calcPath(line);
    }
    else {
      //Same again, but this time we deal with the case of going backwards
      for(int i = start[dim]; i <= finish[dim]; i++) {
	vector<int> link(5);
	link.assign(start,start+4);
	link[dim] = i;
	link[4] = dim;
	line.push_back(link);	
      }
      out = this->calcPath(line);
    }
  }
  return out;
}

double Lattice::W(const int c1[4], const int c2[4])
{
  /*Calculates the loop specified by corners c1 and c2 (which must
    lie in the same plane)*/
  Matrix3cd out = Matrix3cd::Identity();

  //Check that c1 and c2 are on the same plane
  int dim_count = 0;
  for(int i = 0; i < 4; i++) {
    if(c1[i] != c2[i]) {
      dim_count++;
    }
  }
  
  if(dim_count != 2 || c1[0] == c2[0]) {
    cout << "Error! The two corner points do not form a rectangle with two spatial and two temporal sides." << endl;
  }
  else {
    //Get the second corner (going round the loop)
    int c3[4] = {c1[0],c1[1],c1[2],c1[3]};
    c3[0] = c2[0];
    //Calculate the line segments between the first three corners
    out *= this->calcLine(c1,c3);
    out *= this->calcLine(c3,c2);
    //And repeat for the second set of sides
    int c4[4] = {c2[0],c2[1],c2[2],c2[3]};
    c4[0] = c1[0];
    out *= this->calcLine(c2,c4);
    out *= this->calcLine(c4,c1);
  }
  return 1./3 * out.trace().real();
}

double Lattice::W_p(const list cnr1, const list cnr2)
{
  /*Calculates the loop specified by corners c1 and c2 (which must
    lie in the same plane)*/
  int c1[4] = {extract<int>(cnr1[0]),extract<int>(cnr1[1]),extract<int>(cnr1[2]),extract<int>(cnr1[3])};
  int c2[4] = {extract<int>(cnr2[0]),extract<int>(cnr2[1]),extract<int>(cnr2[2]),extract<int>(cnr2[3])};
  return this->W(c1,c2);
}

double Lattice::P(const int site[4],const int mu, const int nu)
{
  /*Calculate the plaquette operator at the given site, on plaquette
    specified by directions mu and nu.*/

  //We define some variables to contain the offsets of the various
  //links around the lattice
  int mu_vec[4] = {0,0,0,0};
  mu_vec[mu] = 1;
  int nu_vec[4] = {0,0,0,0};
  nu_vec[nu] = 1;
  //The links also contain direction information, so we must create a new
  //set of variables
  int link1[5] = {0,0,0,0,mu};
  int link2[5] = {0,0,0,0,nu};
  int link3[5] = {0,0,0,0,mu};
  int link4[5] = {0,0,0,0,nu};

  //Do some assignment
  for(int i = 0; i < 4; i++) {
    link1[i] = site[i];
    link2[i] = site[i] + mu_vec[i];
    link3[i] = site[i] + nu_vec[i];
    link4[i] = site[i];
  }

  //Run through the links and multiply them together.
  Matrix3cd product = this->link(link1);
  product *= this->link(link2);
  product *= this->link(link3).adjoint();
  product *= this->link(link4).adjoint();
  return 1./3 * product.trace().real();
}

double Lattice::P_p(const list site2,const int mu, const int nu)
{
  /*Python wrapper for the plaquette function.*/
  int site[4] = {extract<int>(site2[0]),extract<int>(site2[1]),extract<int>(site2[2]),extract<int>(site2[3])};
  return this->P(site,mu,nu);
}

double Lattice::Si(const int link[5])
{
  /*Calculate the contribution to the action from the given link*/
  int planes[3];
  double Psum = 0;

  //Work out which dimension the link is in, since it'll be irrelevant here
  int j = 0;
  for(int i = 0; i < 4; i++) {
    if(link[4] != i) {
      planes[j] = i;
      j++;
    }
  }

  /*For each plane, calculate the two plaquettes that share the given link*/
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
  //First generate a random matrix whos elements all lie in/on unit circle
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      A(i,j) = double(rand()) / double(RAND_MAX);
      A(i,j) *= exp(2 * lattice::pi * lattice::i * double(rand()) / double(RAND_MAX));
    }
  }
  //Weight the generation by the weighting eps
  Matrix3cd B = Matrix3cd::Identity() + lattice::i * this->eps * A;
  HouseholderQR<Matrix3cd> decomp(B);
  //QR decompose B to make unitary
  Matrix3cd Q = decomp.householderQ();

  //Divide by cube root of determinant to move from U(3) to SU(3)
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
	    //We'll need an array with the link indices
	    int link[5] = {i,j,k,l,m};
	    //Record the old action contribution
	    double Si_old = this->Si(link);
	    //Record the old link in case we need it
	    Matrix3cd linki_old = this->links[i][j][k][l][m];
	    //Get ourselves a random SU3 matrix for the update
	    Matrix3cd randSU3 = this->randSU3s[rand() % this->randSU3s.size()];
	    //Multiply the site
	    this->links[i][j][k][l][m] *= randSU3;
	    //What's the change in the action?
	    double dS = this->Si(link) - Si_old;
	    //Was the change favourable? If not, revert the change
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
  //mu > nu, so there are six plaquettes at each site.
  int nus[6] = {0,0,0,1,1,2};
  int mus[6] = {1,2,3,2,3,3};
  double Ptot = 0;
  //Pretty simple: step through the matrix and add all plaquettes up
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
  //Divide through by number of plaquettes to get mean (simples!)
  return Ptot / (pow(this->n,4) * 6);
}

void Lattice::printL()
{
  /*Print the links out. A bit redundant due to the interfaces library,
   but here in case it's needed.*/
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
  /*Returns the given link as a python nested list. Used in conjunction
   with python interfaces library to extract the links as a nested list
   of numpy matrices.*/
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

//Boost python wrapping of the class
BOOST_PYTHON_MODULE(lattice)
{
  class_<Lattice>("Lattice", init<optional<int,double,int,int,double> >())
    .def("update",&Lattice::update)
    .def("P",&Lattice::P)
    .def("P",&Lattice::P_p)
    .def("Pav",&Lattice::Pav)
    .def("W",&Lattice::W_p)
    .def("printL",&Lattice::printL)
    .def("getLink",&Lattice::getLink)
    .def_readonly("Ncor",&Lattice::Ncor)
    .def_readonly("Ncf",&Lattice::Ncf)
    .def_readonly("n",&Lattice::n);
}
