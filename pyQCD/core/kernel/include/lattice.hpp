#ifndef LATTICE_HPP
#define LATTICE_HPP

#include <cstdlib>
#include <iostream>
#include <ctime>

#include <complex>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>

#include <omp.h>

#include <random.hpp>
#include <linear_operators/linear_operator.hpp>
#include <solvers.hpp>

using namespace Eigen;
using namespace std;
using namespace boost;

typedef vector<Matrix3cd, aligned_allocator<Matrix3cd> > GaugeField;
typedef Triplet<complex<double> > Tlet;

class LinearOperator;

class Lattice
{

public:
  Lattice(const int spatialExtent = 4,
	  const int temporalExtent = 8,
	  const double beta = 5.5,
	  const double ut = 1.0,
	  const double us = 1.0,
	  const double chi = 1.0,
	  const int action = 0,
	  const int nCorrelations = 50,
	  const int updateMethod = 0,
	  const int parallelFlag = 1,
	  const int chunkSize = 4,
	  const int randSeed = -1);
  Lattice(const Lattice& lattice);
  Lattice& operator=(const Lattice& lattice);
  ~Lattice();

  void print();
  Matrix3cd& getLink(const int link[5]);
  Matrix3cd& getLink(const vector<int> link);
  Matrix3cd& getLink(const int index) { return this->links_[index]; };
  void setLink(const int link[5], const Matrix3cd& matrix);

  double us() const { return this->us_; }
  double ut() const { return this->ut_; }
  double chi() const { return this->chi_; }

  int nLinks() const { return this->nLinks_; }
  int nSites() const { return this->nLinks_ / 4; }

  GaugeField getSubLattice(const int startIndex, const int size);

  void metropolis(const int link);
  void metropolisNoStaples(const int link);
  void heatbath(const int link);

  void update();
  void runThreads(const int nUpdates, const int remainder);
  void schwarzUpdate(const int nUpdates);
  void updateSegment(const int startLink, const int nUpdates);

  void thermalize(const int nUpdates);
  void getNextConfig();
  void reunitarize();
  
  Matrix3cd computePath(const vector<vector<int> >& path);
  Matrix3cd computeLine(const int start[4], const int finish[4]);
  double computeWilsonLoop(const int corner1[4], const int corner2[4],
			   const int nSmears = 0,
			   const double smearingParameter = 1.0);
  double computeWilsonLoop(const int corner[4], const int r, const int t,
			   const int dimension, const int nSmears = 0,
			   const double smearingParameter = 1.0);

  double computePlaquette(const int site[4], const int dimension1,
			  const int dimension2);
  double computeRectangle(const int site[4], const int dimension1,
			  const int dimension2);
  double computeTwistedRectangle(const int site[4], const int dimension1,
				 const int dimension2);

  double computeAveragePlaquette();
  double computeAverageRectangle();
  double computeAverageWilsonLoop(const int r, const int t,
				  const int nSmears = 0,
				  const double smearingParameter = 1.0);
  double computeMeanLink();

  double (Lattice::*computeLocalAction)(const int link[5]);
  Matrix3cd (Lattice::*computeStaples)(const int link[5]);
  Matrix3cd makeRandomSu3();
  Matrix2cd makeHeatbathSu2(double coefficients[4],
			    const double weighting);

  Matrix3cd computeQ(const int link[5], const double smearingParameter);
  void smearLinks(const int time, const int nSmears,
		  const double smearingParameter);

  VectorXcd makeSource(const int site[4], const int spin, const int colour,
		       LinearOperator* smearingOperator);
  vector<MatrixXcd> computeWilsonPropagator(
    const double mass, int site[4], const int nSmears,
    const double smearingParameter, const int sourceSmearingType,
    const int nSourceSmears, const double sourceSmearingParameter,
    const int sinkSmearingType, const int nSinkSmears,
    const double sinkSmearingParameter, const int solverMethod,
    const vector<complex<double> >& boundaryConditions, const int precondition,
    const int maxIterations, const double tolerance, const int verbosity);
  vector<MatrixXcd> computePropagator(LinearOperator* diracMatrix,
				      int site[4],
				      const int nSmears,
				      const double smearingParameter,
				      const int sourceSmearingType,
				      const int nSourceSmears,
				      const double sourceSmearingParameter,
				      const int sinkSmearingType,
				      const int nSinkSmears,
				      const double sinkSmearingParameter,
				      const int solverMethod,
				      const int maxIterations,
				      const double tolerance,
				      const int verbosity);
  
  int nCorrelations, spatialExtent, temporalExtent;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
  double beta_, us_, ut_, chi_; // a_s / a_t = chi_
  double anisotropyCoefficients_[4][4];
  double plaquetteTadpoleCoefficients_[4][4];
  double rectangleTadpoleCoefficients_[4][4];
  double twistedRectangleTadpoleCoefficients_[4][4];
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
  Random rng;
  int randSeed_;
};

#endif
