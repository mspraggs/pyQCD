#include <solvers.hpp>

VectorXcd cg(LinearOperator* linop, const VectorXcd& rhs,
	     double& tolerance, int& maxIterations, double& flopRate)
{
  // Perform the conjugate gradient algorithm to solve
  // linop * solution = rhs for solution
  VectorXcd solution = VectorXcd::Zero(rhs.size());

  boost::timer::cpu_timer timer;
  unsigned long long initialFlops = linop->getNumFlops();

  // Use the Hermitian form of the linear operator as that's what CG requires
  VectorXcd r = rhs - linop->applyHermitian(solution); // 2 * N flops
  VectorXcd p = r;

  double oldRes = r.squaredNorm(); // 6 * N + 2 * (N - 1) = 8 * N - 2 flops

  for (int i = 0; i < maxIterations; ++i) {
    VectorXcd linopP = linop->applyHermitian(p);
    complex<double> alpha = oldRes / p.dot(linopP); // 4 + 8 * N flops
    solution = solution + alpha * p; // 8 * N flops
    r = r - alpha * linopP; // 8 * N flops

    double newRes = r.squaredNorm();

    //cout << sqrt(newRes) << endl;

    if (sqrt(newRes) < tolerance) {
      maxIterations = i + 1;
      tolerance = sqrt(newRes);
      break; 
    }
    
    double beta = newRes / oldRes; // 1 flop
    p = r + beta * p; // 8 * N flops
    oldRes = newRes;
  }

  boost::timer::cpu_times const elapsedTimes(timer.elapsed());
  boost::timer::nanosecond_type const elapsed(elapsedTimes.system
					      + elapsedTimes.user);

  unsigned long long totalFlops
    = linop->getNumFlops() - initialFlops + 42 * rhs.size() + 3;

  flopRate = (double) totalFlops / elapsed * 1000.0;

  return linop->undoHermiticity(solution);
}



VectorXcd bicgstab(LinearOperator* linop, const VectorXcd& rhs,
		   double& tolerance, int& maxIterations, double& flopRate)
{
  // Perform the biconjugate gradient stabilized algorithm to
  // solve linop * solution = rhs for solution
  VectorXcd solution = VectorXcd::Zero(rhs.size());

  boost::timer::cpu_timer timer;
  
  // Use the normal for of the linear operator as there's no requirement
  // for the linop to be hermitian
  unsigned long long initialFlops = linop->getNumFlops();

  VectorXcd r = rhs - linop->apply(solution); // 2 * N flops
  VectorXcd rhat = r;
  double rhatNorm = r.squaredNorm(); // 6 * N + 2 * (N - 1) = 8 * N - 2 flops
  
  complex<double> rho(1.0, 0.0);
  complex<double> alpha(1.0, 0.0);
  complex<double> omega(1.0, 0.0);

  VectorXcd p = VectorXcd::Zero(rhs.size());
  VectorXcd v = VectorXcd::Zero(rhs.size());

  double residual = r.squaredNorm(); // 6 * N + 2 * (N - 1) = 8 * N - 2 flops

  for (int i = 0; i < maxIterations; ++i) {
    // 6 * N + 2 * (N - 1) = 8 * N - 2 flops
    complex<double> newRho = rhat.dot(r);
    if (abs(newRho) == 0.0) {
      maxIterations = i;
      tolerance = sqrt(residual / rhatNorm);
      break;
    }
    complex<double> beta = (newRho / rho) * (alpha / omega); // 24 flops
    
    p = r + beta * (p - omega * v); // N * 16
    v = linop->apply(p);

    alpha = newRho / rhat.dot(v); // 6 + 8 * N - 2 = 4 + 8 * N flops
    VectorXcd s = r - alpha * v; // 8 * N flops
    VectorXcd t = linop->apply(s);

    omega = t.dot(s) / t.squaredNorm(); // 6 + 16 * N - 4 = 16 * N + 2 flops
    solution += alpha * p + omega * s; // 14 * N flops

    r = s - omega * t; // 8 * N flops

    residual = r.squaredNorm(); // 8 * N - 2 flops
    
    if (sqrt(residual / rhatNorm) < tolerance) {
      maxIterations = i + 1;
      tolerance = sqrt(residual / rhatNorm);
      break;
    }

    rho = newRho;
  }

  boost::timer::cpu_times const elapsedTimes(timer.elapsed());
  boost::timer::nanosecond_type const elapsed(elapsedTimes.system
					      + elapsedTimes.user);

  long long totalFlops = linop->getNumFlops() - initialFlops
    + 96 * rhs.size() + 22;

  flopRate = ((double) totalFlops) / elapsed * 1000.0;

  return solution;
}
