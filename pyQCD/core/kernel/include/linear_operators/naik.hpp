#ifndef NAIK_HPP
#define NAIK_HPP

#include <Eigen/Dense>

#include <complex>

#include <omp.h>

#include <lattice.hpp>
#include <utils.hpp>
#include <linear_operators/linear_operator.hpp>
#include <linear_operators/hopping_term.hpp>

using namespace Eigen;
using namespace std;

class Naik : public LinearOperator
{
  // Basic unpreconditioned Naik Dirac operator

public:
  Naik(const double mass,
       const vector<complex<double> >& boundaryConditions,
       Lattice* lattice);
  ~Naik();

  VectorXcd apply(const VectorXcd& psi);
  VectorXcd applyHermitian(const VectorXcd& psi);
  VectorXcd makeHermitian(const VectorXcd& psi);

private:
  // Pointer to the lattice object containing the gauge links
  Lattice* lattice_;
  HoppingTerm* nearestNeighbour_; // Our Wilson hopping piece
  HoppingTerm* nextNextNearestNeighbour_; // Our Naik hopping piece
  double mass_; // Mass of the Naik fermion
  vector<vector<complex<double> > > boundaryConditions_;
};

#endif
