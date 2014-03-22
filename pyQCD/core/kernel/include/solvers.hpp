#ifndef SOLVERS_HPP
#define SOLVERS_HPP

#include <Eigen/Dense>

#include <omp.h>
#include <iostream>

#include <linear_operators/linear_operator.hpp>

using namespace Eigen;
using namespace std;

VectorXcd cg(LinearOperator* linop, const VectorXcd& rhs,
	     const double tolerance, const int maxIterations,
	     double& finalResidual, int& totalIterations);

VectorXcd bicgstab(LinearOperator* linop, const VectorXcd& rhs,
		   const double tolerance, const int maxIterations,
		   double& finalResidual, int& totalIterations);

#endif
