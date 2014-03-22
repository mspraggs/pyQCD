#include <solvers.hpp>

VectorXcd cg(LinearOperator* linop, const VectorXcd& rhs,
	     const double tolerance, const int maxIterations,
	     double& finalResidual, int& totalIterations)
{
  // Perform the conjugate gradient algorithm to solve
  // linop * solution = rhs for solution
  VectorXcd solution = VectorXcd::Zero(rhs.size());

  // Use the Hermitian form of the linear operator as that's what CG requires
  VectorXcd r = rhs - linop->applyHermitian(solution);
  VectorXcd p = r;

  double oldRes = r.squaredNorm();

  for (int i = 0; i < maxIterations; ++i) {
    VectorXcd linopP = linop->applyHermitian(p);
    double alpha = oldRes / p.dot(linopP).real();
    solution = solution + alpha * p;
    r = r - alpha * linopP;

    double newRes = r.squaredNorm();

    //cout << sqrt(newRes) << endl;

    if (sqrt(newRes) < tolerance) {
      totalIterations = i + 1;
      finalResidual = sqrt(newRes);
      break; 
    }
    
    double beta = newRes / oldRes;
    p = r + beta * p;
    oldRes = newRes;
  }

  return linop->undoHermiticity(solution);
}



VectorXcd bicgstab(LinearOperator* linop, const VectorXcd& rhs,
		   const double tolerance, const int maxIterations,
		   double& finalResidual, int& totalIterations)
{
  // Perform the biconjugate gradient stabilized algorithm to
  // solve linop * solution = rhs for solution
  VectorXcd solution = VectorXcd::Zero(rhs.size());
  
  // Use the normal for of the linear operator as there's no requirement
  // for the linop to be hermitian
  VectorXcd r = rhs - linop->apply(solution);
  VectorXcd rhat = r;
  double rhatNorm = r.squaredNorm();
  
  complex<double> rho(1.0, 0.0);
  complex<double> alpha(1.0, 0.0);
  complex<double> omega(1.0, 0.0);

  VectorXcd p = VectorXcd::Zero(rhs.size());
  VectorXcd v = VectorXcd::Zero(rhs.size());

  double residual = r.squaredNorm();

  for (int i = 0; i < maxIterations; ++i) {
    complex<double> newRho = rhat.dot(r);
    if (abs(newRho) == 0.0) {
      totalIterations = i;
      finalResidual = sqrt(residual / rhatNorm);
      break;
    }
    complex<double> beta = (newRho / rho) * (alpha / omega);
    
    p = r + beta * (p - omega * v);
    v = linop->apply(p);

    alpha = newRho / rhat.dot(v);
    VectorXcd s = r - alpha * v;
    VectorXcd t = linop->apply(s);

    omega = t.dot(s) / t.squaredNorm();
    solution += alpha * p + omega * s;

    r = s - omega * t;

    residual = r.squaredNorm();
    
    if (sqrt(residual / rhatNorm) < tolerance) {
      totalIterations = i + 1;
      finalResidual = sqrt(residual / rhatNorm);
      break;
    }

    rho = newRho;
  }

  return solution;
}
