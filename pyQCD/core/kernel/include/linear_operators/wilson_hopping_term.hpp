#ifndef WILSON_HOPPING_TERM_HPP
#define WILSON_HOPPING_TERM_HPP

#include <Eigen/Dense>

#include <complex>

#include <omp.h>

#include <lattice.hpp>
#include <utils.hpp>
#include <linear_operators/linear_operator.hpp>

using namespace Eigen;
using namespace std;

class WilsonHoppingTerm : public LinearOperator
{
  // Basic unpreconditioned Wilson Dirac operator

public:
  WilsonHoppingTerm(const vector<complex<double> >& boundaryConditions,
		    Lattice* lattice);
  WilsonHoppingTerm(const vector<complex<double> >& boundaryConditions,
		    const vector<Matrix4cd>& spinStructures,
		    Lattice* lattice);
  WilsonHoppingTerm(const vector<complex<double> >& boundaryConditions,
		    const Matrix4cd& spinStructure,
		    Lattice* lattice);
  ~WilsonHoppingTerm();

  VectorXcd multiplyGamma5(const VectorXcd& psi);

  VectorXcd apply3d(const VectorXcd& psi);
  VectorXcd apply(const VectorXcd& psi);
  VectorXcd applyHermitian(const VectorXcd& psi);
  VectorXcd undoHermiticity(const VectorXcd& psi);

private:
  // Pointer to the lattice object containing the gauge links
  Lattice* lattice_;
  int operatorSize_; // Size of vectors on which the operator may operate
  // The spin matrices required by the operator
  vector<Matrix4cd, aligned_allocator<Matrix4cd> > spinStructures_;
  // Nearest neighbour indices
  vector<vector<int> > nearestNeighbours_;
  vector<vector<complex<double> > > boundaryConditions_;
  double tadpoleCoefficients_[4];
};

#endif
