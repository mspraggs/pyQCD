#ifndef DWF_HPP
#define DWF_HPP

#include <Eigen/Dense>

#include <complex>

#include <omp.h>

#include <lattice.hpp>
#include <utils.hpp>
#include <linear_operators/linear_operator.hpp>
#include <linear_operators/wilson.hpp>
#include <linear_operators/hamber_wu.hpp>
#include <linear_operators/naik.hpp>

using namespace Eigen;
using namespace std;

class DWF : public LinearOperator
{
  // Generic Shamir DWF operator that uses any kernel that has quark
  // mass as the only parameter.

public:
  DWF(const double mass, const double M5, const int Ls,
      const int kernelType,
      const vector<complex<double> >& boundaryConditions,
      Lattice* lattice);
  ~DWF();

  VectorXcd apply(const VectorXcd& psi);
  VectorXcd applyHermitian(const VectorXcd& psi);
  VectorXcd makeHermitian(const VectorXcd& psi);

private:
  // Pointer to the lattice object containing the gauge links
  Lattice* lattice_;
  LinearOperator* kernel_; // The 4D kernel
  double mass_; // Mass of the Wilson fermion
  int Ls_;
  vector<vector<complex<double> > > boundaryConditions_;
};

#endif
