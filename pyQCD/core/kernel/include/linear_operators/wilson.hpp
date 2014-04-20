#ifndef WILSON_HPP
#define WILSON_HPP

#include <Eigen/Dense>

#include <complex>

#include <omp.h>

#include <lattice.hpp>
#include <utils.hpp>
#include <linear_operators/linear_operator.hpp>
#include <linear_operators/hopping_term.hpp>

using namespace Eigen;
using namespace std;

class Wilson : public LinearOperator
{
  // Basic unpreconditioned Wilson Dirac operator

public:
  Wilson(const double mass,
	 const vector<complex<double> >& boundaryConditions,
	 Lattice* lattice);
  ~Wilson();

  VectorXcd apply(const VectorXcd& psi);
  VectorXcd applyHermitian(const VectorXcd& psi);
  VectorXcd makeHermitian(const VectorXcd& psi);

  complex<double> lowerRowDot(const VectorXcd& psi, const int row);
  complex<double> upperRowDot(const VectorXcd& psi, const int row);
  VectorXcd applyDiagonal(const VectorXcd& psi);

private:
  // Pointer to the lattice object containing the gauge links
  Lattice* lattice_;
  HoppingTerm* hoppingMatrix_; // This operator does the derivative
  double mass_; // Mass of the Wilson fermion
  vector<vector<complex<double> > > boundaryConditions_;
};

#endif
