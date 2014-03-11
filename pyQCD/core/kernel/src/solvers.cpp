#include <solvers.hpp>

VectorXcd cg(LinearOperator* linop, const VectorXcd& lhs,
	     const double tolerance, const int maxIterations)
{
  // Perform the conjugate gradient algorithm to solve
  // linop * solution = lhs for solution
  VectorXcd solution = VectorXcd::Zero(lhs.size());

  // Use the Hermitian form of the linear operator as that's what CG requires
  VectorXcd r = lhs - linop->applyHermitian(solution);
  VectorXcd p = r;

  double oldRes = r.squaredNorm();

  for (int i = 0; i < maxIterations; ++i) {
    VectorXcd linopP = linop->applyHermitian(p);
    double alpha = oldRes / p.dot(linopP).real();
    solution = solution + alpha * p;
    r = r - alpha * linopP;

    double newRes = r.squaredNorm();

    //cout << sqrt(newRes) << endl;

    if (sqrt(newRes) < tolerance)
      break;
    
    double beta = newRes / oldRes;
    p = r + beta * p;
    oldRes = newRes;
  }

  return linop->undoHermiticity(solution);
}
