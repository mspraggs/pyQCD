#ifndef LATTICE_HPP
#define LATTICE_HPP

#include <cstdlib>
#include <iostream>
#include <ctime>

#include <complex>
#include <vector>

#include <boost/random.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>

#include <omp.h>

using namespace Eigen;
using namespace std;
using namespace boost;

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
	  const double u0 = 1.0,
	  const int action = 0,
	  const int nCorrelations = 50,
	  const double rho = 0.3,
	  const double epsilon = 0.24);

  Lattice(const Lattice& lattice);
  ~Lattice();

  void print();
  Matrix3cd& getLink(const int link[5]);

  void runThreads(const int chunkSize, const int nUpdates,
		  const int remainder);
  void schwarzUpdate(const int chunkSize, const int nUpdates);
  void update();
  void updateSegment(const int n0, const int n1, const int n2,
		     const int n3, const int chunkSize, const int nUpdates);

  void heatbath();

  void thermalize();
  void getNextConfig();
  
  void computePath(const vector<vector<int> >& path, Matrix3cd& out);
  void computeLine(const int start[4], const int finish[4],
		   Matrix3cd& out);
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

  double computeAveragePlaquette();
  double computeAverageRectangle();
  double computeAverageWilsonLoop(const int r, const int t,
			      const int nSmears = 0);
  double computeMeanLink();

  double (Lattice::*computeLocalAction)(const int link[5]);
  void (Lattice::*computeStaples)(const int link[5], Matrix3cd& out);
  void makeRandomSu3(Matrix3cd& out);
  void makeHeatbathSu2(Matrix2cd& out, double coefficients[4],
		       const double weighting);

  void computeQ(const int link[5], Matrix3cd& out);
  void smearLinks(const int time, const int nSmears);

  SparseMatrix<complex<double> > computeDiracMatrix(const double mass,
						    const double spacing);
  VectorXcd computePropagator(const double mass, int site[4],
			      const int lorentzIndex, const int colourIndex,
			      const double spacing);

  int nCorrelations, nEdgePoints;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
  double beta_, epsilon_, u0_, rho_;
  int nUpdates_, action_;
  double computeLocalWilsonAction(const int link[5]);
  double computeLocalRectangleAction(const int link[5]);
  double computeLocalTwistedRectangleAction(const int link[5]);
  void computeWilsonStaples(const int link[5], Matrix3cd& out);
  void computeRectangleStaples(const int link[5], Matrix3cd& out);
  void computeTwistedRectangleStaples(const int link[5], Matrix3cd& out);
  GaugeField links_;
  Sub4Field randSu3s_;
};

#endif
