#ifndef LATTICE_HPP
#define LATTICE_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <complex>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <unsupported/Eigen/MatrixFunctions>
#include "gil.cpp"
#include <omp.h>

using namespace Eigen;
namespace bst = boost;
using namespace std;

typedef vector<Matrix3cd, aligned_allocator<Matrix3cd> > Sub4Field;
typedef vector<Sub4Field> Sub3Field;
typedef vector<Sub3Field> Sub2Field;
typedef vector<Sub2Field> SubField;
typedef vector<SubField> GaugeField;
typedef Triplet<complex<double> > Tlet;

class Lattice
{

public:
  Lattice(const int n = 8,
	  const double beta = 5.5,
	  const int Ncor = 50,
	  const int Ncf = 1000,
	  const double eps = 0.24,
	  const double a = 0.25,
	  const double rho = 0.3,
	  const double u0 = 1,
	  const int action = 0);
  Lattice(const Lattice& L);
  double init_u0();

  ~Lattice();
  Matrix3cd calcPath(const vector<vector<int> > path);
  Matrix3cd calcLine(const int start[4], const int finish[4]);
  double calcWilsonLoop(const int c1[4], const int c2[4],
			const int n_smears = 0);
  double calcWilsonLoop(const int c[4], const int r, const int t,
			const int dim, const int n_smears = 0);
  double calcAverageWilson(const int r, const int t, const int n_smears = 0);
  double calcPlaquette(const int site[4], const int mu, const int nu);
  double calcRectangle(const int site[4], const int mu, const int nu);
  double calcTwistRect(const int site[4],const int mu, const int nu);
  double calcAveragePlaq();
  double calcAverageRect();
  double (Lattice::*calcLocalAction)(const int link[5]);
  Matrix3cd makeRandomSU3();
  void thermalize();
  void getNextConfig();
  void runThreads(const int size, const int n_updates, const int remainder);
  void schwarzUpdate(const int size, const int n_times);
  void updateLattice();
  void updateSegment(const int i, const int j, const int k,
		     const int l, const int size, const int n_updates);
  void printLattice();
  Matrix3cd getLink(const int link[5]);
  void smearLinks(const int time, const int n_smears);
  Matrix3cd calcQ(const int link[5]);

  SparseMatrix<complex<double> > calcDiracMatrix(const double mass);
  VectorXcd calcPropagator(const double mass, int site[4],
			   const int alpha, const int a);

  int Ncor, Ncf, n;

protected:
  double beta, eps, a, u0, smear_eps;
  int nUpdates, action;
  double calcLocalWilson(const int link[5]);
  double calcLocalRectangle(const int link[5]);
  double calcLocalTRectangle(const int link[5]);
  GaugeField links;
  Sub4Field randSU3s;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
};

#endif
