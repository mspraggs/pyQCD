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

typedef vector<Matrix3cd, aligned_allocator<Matrix3cd> > GaugeField;
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
	  const double epsilon = 0.24, 
	  const int updateMethod = 0,
	  const int parallelFlag = 1);
  Lattice(const Lattice& lattice);
  ~Lattice();

  void convertIndex(const int index, int link[5]);

  void print();
  Matrix3cd& getLink(const int link[5]);
  Matrix3cd& getLink(const vector<int> link);

  GaugeField getSubLattice(const int startIndex, const int size);

  void monteCarlo(const int link);
  void monteCarloNoStaples(const int link);
  void heatbath(const int link);

  void update();
  void runThreads(const int nUpdates, const int remainder);
  void schwarzUpdate(const int nUpdates);
  void updateSegment(const int startLink, const int nUpdates);

  void thermalize();
  void getNextConfig();
  
  Matrix3cd computePath(const vector<vector<int> >& path);
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

  double computeAveragePlaquette();
  double computeAverageRectangle();
  double computeAverageWilsonLoop(const int r, const int t,
			      const int nSmears = 0);
  double computeMeanLink();

  double (Lattice::*computeLocalAction)(const int link[5]);
  Matrix3cd (Lattice::*computeStaples)(const int link[5]);
  Matrix3cd makeRandomSu3();
  Matrix2cd makeHeatbathSu2(double coefficients[4],
		       const double weighting);

  Matrix3cd computeQ(const int link[5]);
  void smearLinks(const int time, const int nSmears);

  SparseMatrix<complex<double> > computeDiracMatrix(const double mass,
						    const double spacing);
  MatrixXcd computePropagator(const double mass, int site[4],
			      const double spacing);
  MatrixXcd computePropagator(const double mass, int site[4],
			      const double spacing,
			      const SparseMatrix<complex<double> >& D);
  MatrixXcd computeZeroMomPropagator(const double mass, const int time,
				     const double spacing);
  vector<MatrixXcd> computePropagators(const double mass, const double spacing);

  int nCorrelations, nEdgePoints;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
  double beta_, epsilon_, u0_, rho_;
  int nUpdates_, action_, updateMethod_, parallelFlag_, nLinks_;
  double computeLocalWilsonAction(const int link[5]);
  double computeLocalRectangleAction(const int link[5]);
  double computeLocalTwistedRectangleAction(const int link[5]);
  Matrix3cd computeWilsonStaples(const int link[5]);
  Matrix3cd computeRectangleStaples(const int link[5]);
  Matrix3cd computeTwistedRectangleStaples(const int link[5]);
  void (Lattice::*updateFunction_)(const int link);
  GaugeField links_;
  GaugeField randSu3s_;
  vector<int> chunkSequence_;
  vector<int> evenBlocks_;
  vector<int> oddBlocks_;
  vector<vector<vector<int> > > propagatorColumns_;
};

#endif
