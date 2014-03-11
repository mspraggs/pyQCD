#ifndef UNPRECONDITIONED_WILSON_HPP
#define UNPRECONDITIONED_WILSON_HPP

#include <Eigen/Dense>

#include <complex>

#include <omp.h>

#include <lattice.hpp>
#include <utils.hpp>
#include <fermion_actions/linear_operator.hpp>

using namespace Eigen;
using namespace std;

class UnpreconditionedWilson : public LinearOperator
{
  // Basic unpreconditioned Wilson Dirac operator

public:
  UnpreconditionedWilson(const double mass, Lattice* lattice);
  ~UnpreconditionedWilson();

  VectorXcd apply(const VectorXcd& psi);
  VectorXcd applyHermitian(const VectorXcd& psi);
  VectorXcd undoHermiticity(const VectorXcd& psi);

private:
  // Pointer to the lattice object containing the gauge links
  Lattice* lattice_;
  double mass_; // Mass of the Wilson fermion
  int operatorSize_; // Size of vectors on which the operator may operate
  // The 1 +/- gamma_mu matrices required by the operator
  vector<Matrix4cd, aligned_allocator<Matrix4cd> > spinStructures_;
  // The spinStructes_ above multiplied by gamma_5 from the right
  vector<Matrix4cd, aligned_allocator<Matrix4cd> > hermitianSpinStructures_;
};

#endif
