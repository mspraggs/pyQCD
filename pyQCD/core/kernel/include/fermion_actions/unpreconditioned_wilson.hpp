#ifndef UNPRECONDITIONED_WILSON_HPP
#define UNPRECONDITIONED_WILSON_HPP

#include <Eigen/Dense>

#include <omp.h>

#include <lattice.hpp>
#include <utils.hpp>
#include "linear_operator.hpp"

using namespace Eigen;
using namespace std;

class UnpreconditionedWilson : public LinearOperator
{
  // Basic unpreconditioned Wilson Dirac operator

public:
  UnpreconditionedWilson(const double mass, const Lattice* lattice);
  ~UnpreconditionedWilson();

  VectorXcd apply(const VectorXcd& x);
  VectorXcd applyHermitian(const VectorXcd& x);
  VectorXcd undoHermiticity(const VectorXcd& x);

private:
  Lattice* lattice_; // Pointer to the lattice object containing the gauge links
  double mass_; // Mass of the Wilson fermion
  int operatorSize_; // Size of vectors on which the operator may operate
  // The 1 +/- gamma_mu matrices required by the operator
  vector<Matrix4cd, aligned_allocator<Matrix4cd> > spinStructures_;
  // The spinStructes_ above multiplied by gamma_5 from the right
  vector<Matrix4cd, aligned_allocator<Matrix4cd> > hermitianSpinStructures_;
};
