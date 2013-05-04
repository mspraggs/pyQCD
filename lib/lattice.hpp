#ifndef LATTICE_HPP
#define LATTICE_HPP

#include <cstdlib>
#include <iostream>
#include <ctime>

#include <complex>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>
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
  Lattice(const int nEdgePoints = 8,
	  const double beta = 5.5,
	  const int nCorrelations = 50,
	  const int nConfigurations = 1000,
	  const double eps = 0.24,
	  const double a = 0.25,
	  const double rho = 0.3,
	  const double u0 = 1,
	  const int action = 0);

  Lattice(const Lattice& lattice);
  ~Lattice();

  void printLattice();
  Matrix3cd getLink(const int link[5]);

  void runThreads(const int chunkSize, const int nUpdates,
		  const int remainder);
  void schwarzUpdate(const int chunkSize, const int nUpdates);
  void update();
  void updateSegment(const int n0, const int n1, const int n2,
		     const int n3, const int chunkSize, const int nUpdates);

  void thermalize();
  void getNextConfig();
  
  Matrix3cd computePath(const vector<vector<int> > path);
  Matrix3cd computeLine(const int start[4], const int finish[4]);
  double computeWilsonLoop(const int corner1[4], const int corner2[4],
			   const int nSmears = 0);
  double computeWilsonLoop(const int corner[4], const int r, const int t,
			   const int dimension, const int nSmears = 0);

  double computePlaquette(const int site[4], const int dimension1,
			  const int dimension2);
  double computeRectangle(const int site[4], const int dimension1,
			  const int dimension2);
  double computeTwistedRectangle(const int site[4], const int dimension1,
				 const int dimension2);
  double computeAveragePlaq();
  double computeAverageRect();
  double computeAverageWilson(const int r, const int t,
			      const int nSmears = 0);
  double (Lattice::*computeLocalAction)(const int link[5]);
  Matrix3cd makeRandomSu3();

  Matrix3cd computeQ(const int link[5]);
  void smearLinks(const int time, const int nSmears);

  SparseMatrix<complex<double> > computeDiracMatrix(const double mass);
  VectorXcd computePropagator(const double mass, int site[4],
			      const int lorentzIndex, const int colourIndex);

  int nCorrelations, nConfigurations, nEdgePoints;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
  double beta_, eps_, a_, u0_, rho_;
  int nUpdates_, action_;
  double computeLocalWilson(const int link[5]);
  double computeLocalRectangle(const int link[5]);
  double computeLocalTRectangle(const int link[5]);
  GaugeField links_;
  Sub4Field randSu3s_;
};

#endif
