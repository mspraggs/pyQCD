#ifndef HAMBER_WU_HPP
#define HAMBER_WU_HPP

#include <Eigen/Dense>

#include <complex>

#include <omp.h>

#include <lattice.hpp>
#include <utils.hpp>
#include <linear_operators/linear_operator.hpp>
#include <linear_operators/hopping_term.hpp>

using namespace Eigen;
using namespace std;

class HamberWu : public LinearOperator
{
  // Basic unpreconditioned HamberWu Dirac operator

public:
  HamberWu(const double mass,
	   const vector<complex<double> >& boundaryConditions,
	   const Lattice* lattice);
  ~HamberWu();

  VectorXcd apply(const VectorXcd& psi);
  VectorXcd applyHermitian(const VectorXcd& psi);
  VectorXcd makeHermitian(const VectorXcd& psi);

private:
  // Pointer to the lattice object containing the gauge links
  const Lattice* lattice_;
  HoppingTerm* nearestNeighbour_; // Our Wilson hopping piece
  HoppingTerm* nextNearestNeighbour_; // Our Hamber-Wu hopping piece
  double mass_; // Mass of the HamberWu fermion
  vector<vector<complex<double> > > boundaryConditions_;
};

#endif
