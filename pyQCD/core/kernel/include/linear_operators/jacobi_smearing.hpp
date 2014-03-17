#ifndef JACOBI_SMEARING_HPP
#define JACOBI_SMEARING_HPP

#include <Eigen/Dense>

#include <complex>

#include <omp.h>

#include <lattice.hpp>
#include <utils.hpp>
#include <linear_operators/linear_operator.hpp>

using namespace Eigen;
using namespace std;

class JacobiSmearing : public LinearOperator
{
  // Basic unpreconditioned Wilson Dirac operator

public:
  JacobiSmearing(const int numSmears, const double smearingParameter,
		 const vector<complex<double> >& boundaryConditions,
		 Lattice* lattice);
  ~JacobiSmearing();

  VectorXcd apply(const VectorXcd& psi);
  VectorXcd applyOnce(const VectorXcd& psi);
  VectorXcd applyHermitian(const VectorXcd& psi);
  VectorXcd undoHermiticity(const VectorXcd& psi);

private:
  // Pointer to the lattice object containing the gauge links
  Lattice* lattice_;
  int operatorSize_; // Size of vectors on which the operator may operate
  // We'll need the identity matrix when applying the operator
  Matrix4cd identity_;
  vector<vector<int> > nearestNeighbours_;
  vector<vector<complex<double> > > boundaryConditions_;
  int numSmears_;
  double smearingParameter_;
  double tadpoleCoefficients_[4];
};

#endif
