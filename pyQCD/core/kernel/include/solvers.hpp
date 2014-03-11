#ifndef SOLVERS_HPP
#define SOLVERS_HPP

#include <Eigen/Dense>

#include <omp.h>
#include <iostream>

#include <fermion_actions/linear_operator.hpp>

using namespace Eigen;
using namespace std;

VectorXcd cg(const LinearOperator* linop, const VectorXcd& lhs,
	     const double tolerance, const int maxIterations);

#endif
