#ifndef LINOP_UTILS_HPP
#define LINOP_UTILS_HPP

#include <Eigen/Dense>

#include <lattice.hpp>
#include <utils.hpp>

using namespace std;
using namespace Eigen;

class Lattice;

namespace pyQCD
{
  VectorXcd multiplyGamma5(const VectorXcd& psi);

  vector<vector<int> > getNeighbourIndices(const int hopSize, Lattice* lattice);

  vector<vector<complex<double> > > getBoundaryConditions(
    const int hopSize, const vector<complex<double> >& boundaryConditions,
    Lattice* lattice);
}

#endif
