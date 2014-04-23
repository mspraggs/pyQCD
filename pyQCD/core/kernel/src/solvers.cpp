#include <solvers.hpp>

void arnoldi(MatrixXcd& V, MatrixXcd& H, LinearOperator* linop,
	     const VectorXcd& rhs, const int numIterations,
	     const int precondition)
{
  // Runs the Arnoldi relation to find an orthonormal basis of the Krylov
  // subspace K(A, b)

  double beta = rhs.norm();

  V = MatrixXcd::Zero(rhs.size(), numIterations + 1);
  H = MatrixXcd::Zero(numIterations + 1, numIterations);
  V.col(0) = rhs / beta;

  for (int i = 1; i < numIterations + 1; ++i) {
    VectorXcd q;
    if (precondition == 1)
      q = linop->applyPreconditioned(V.col(i - 1));
    else
      q = linop->apply(V.col(i - 1));

    // Here we're basically doing Gram-Schmidt to make q orthogonal to all
    // V(0), V(1), ... , V(i-1)
    for (int j = 0; j < i; ++j) {
      H(j, i - 1) = q.dot(V.col(j));
      q -= H(j, i - 1) * V.col(j);
    }

    H(i, i - 1) = q.norm();
    V.col(i) = q / H(i, i - 1);
  }
}

VectorXcd cg(LinearOperator* linop, const VectorXcd& rhs,
	     double& tolerance, int& maxIterations, double& time,
	     const int precondition)
{
  // Perform the conjugate gradient algorithm to solve
  // linop * solution = rhs for solution

  int N = rhs.size() / 2;

  VectorXcd rhsHermitian;
  VectorXcd rhsOdd;
  VectorXcd oddSolution;
  VectorXcd solution = VectorXcd::Zero(rhs.size());

  if (precondition == 1) {
    rhsHermitian = linop->makeEvenOddSource(rhs);
    rhsOdd = linop->makeHermitian(rhsHermitian.tail(N));
    solution.head(N) = linop->applyEvenEvenInv(rhsHermitian.head(N));
    oddSolution = VectorXcd::Zero(N);
  }
  else {
    rhsHermitian = linop->makeHermitian(rhs);
  }
        

  boost::timer::cpu_timer timer;

  // Use the Hermitian form of the linear operator as that's what CG requires
  VectorXcd r 
    = (precondition == 1)
    ? rhsOdd - linop->applyPreconditionedHermitian(oddSolution)
    : rhsHermitian - linop->applyHermitian(solution); // 2 * N flops
  VectorXcd p = r;

  double oldRes = r.squaredNorm(); // 6 * N + 2 * (N - 1) = 8 * N - 2 flops

  for (int i = 0; i < maxIterations; ++i) {
    VectorXcd linopP
      = (precondition == 1)
      ? linop->applyPreconditionedHermitian(p)
      : linop->applyHermitian(p);
    complex<double> alpha = oldRes / p.dot(linopP); // 4 + 8 * N flops
    if (precondition == 1)
      oddSolution += alpha * p; // 8 * N flops
    else
      solution += alpha * p;
    r -= alpha * linopP; // 8 * N flops

    double newRes = r.squaredNorm();

    if (sqrt(newRes) < tolerance) {
      maxIterations = i + 1;
      oldRes = newRes;
      break; 
    }
    
    double beta = newRes / oldRes; // 1 flop
    p = r + beta * p; // 8 * N flops
    oldRes = newRes;
  }

  tolerance = sqrt(oldRes);

  boost::timer::cpu_times const elapsedTimes(timer.elapsed());
  boost::timer::nanosecond_type const elapsed(elapsedTimes.system
					      + elapsedTimes.user);

  time = elapsed / 1.0e9;

  if (precondition == 1) {
    solution.tail(N) = oddSolution;
    solution = linop->makeEvenOddSolution(solution);
  }

  return solution;
}



VectorXcd bicgstab(LinearOperator* linop, const VectorXcd& rhs,
		   double& tolerance, int& maxIterations, double& time,
		   const int precondition)
{
  // Perform the biconjugate gradient stabilized algorithm to
  // solve linop * solution = rhs for solution
  VectorXcd solution = VectorXcd::Zero(rhs.size());

  int N = (precondition == 1) ? rhs.size() / 2 : rhs.size();

  VectorXcd rhsOdd;
  VectorXcd oddSolution;

  if (precondition == 1) {
    VectorXcd rhsEvenOdd = linop->makeEvenOddSource(rhs);
    rhsOdd = rhsEvenOdd.tail(N);
    solution.head(N) = linop->applyEvenEvenInv(rhsEvenOdd.head(N));
    oddSolution = VectorXcd::Zero(N);
  }

  boost::timer::cpu_timer timer;
  
  // Use the normal for of the linear operator as there's no requirement
  // for the linop to be hermitian
  
  VectorXcd r 
    = (precondition == 1)
    ? rhsOdd - linop->applyPreconditioned(oddSolution)
    : rhs - linop->apply(solution); // 2 * N flops
  VectorXcd r0 = r;
  double r0Norm = r0.squaredNorm(); // 6 * N + 2 * (N - 1) = 8 * N - 2 flops
  
  complex<double> rho(1.0, 0.0);
  complex<double> alpha(1.0, 0.0);
  complex<double> omega(1.0, 0.0);

  VectorXcd p = VectorXcd::Zero(N);
  VectorXcd v = VectorXcd::Zero(N);
  VectorXcd s = VectorXcd::Zero(N);
  VectorXcd t = VectorXcd::Zero(N);

  double residual = r.squaredNorm(); // 6 * N + 2 * (N - 1) = 8 * N - 2 flops

  for (int i = 0; i < maxIterations; ++i) {
    // 6 * N + 2 * (N - 1) = 8 * N - 2 flops
    complex<double> rhoOld = rho;
    rho = r0.dot(r);

    if (abs(rho) == 0.0) {
      maxIterations = i;
      tolerance = sqrt(residual / r0Norm);
      break;
    }
    complex<double> beta = (rho / rhoOld) * (alpha / omega); // 24 flops
    
    p = r + beta * (p - omega * v); // N * 16
    v = (precondition == 1) ? linop->applyPreconditioned(p) : linop->apply(p);

    alpha = rho / r0.dot(v); // 6 + 8 * N - 2 = 4 + 8 * N flops
    s = r - alpha * v; // 8 * N flops
    t = (precondition == 1) ? linop->applyPreconditioned(s) : linop->apply(s);

    omega = t.dot(s) / t.squaredNorm(); // 6 + 16 * N - 4 = 16 * N + 2 flops
    if (precondition == 1)
      oddSolution += alpha * p + omega * s; // 14 * N flops
    else
      solution += alpha * p + omega * s; // 14 * N flops

    residual = r.squaredNorm(); // 8 * N - 2 flops

    r = s - omega * t; // 8 * N flops
    
    if (sqrt(residual / r0Norm) < tolerance) {
      maxIterations = i + 1;
      break;
    }
  }

  tolerance = sqrt(residual / r0Norm);  

  boost::timer::cpu_times const elapsedTimes(timer.elapsed());
  boost::timer::nanosecond_type const elapsed(elapsedTimes.system
					      + elapsedTimes.user);

  time = elapsed / 1.0e9;

  if (precondition == 1) {
    solution.tail(N) = oddSolution;
    solution = linop->makeEvenOddSolution(solution);
  }

  return solution;
};

VectorXcd gmres(LinearOperator* linop, const VectorXcd& rhs,
		double& tolerance, int& maxIterations, double& time,
		const int precondition)
{
  // Here we do the restarted generalized minimal residual method

  VectorXcd solution = VectorXcd::Zero(rhs.size());

  int N = rhs.size() / 2;

  VectorXcd rhsOdd;
  VectorXcd oddSolution;

  if (precondition == 1) {
    VectorXcd rhsEvenOdd = linop->makeEvenOddSource(rhs);
    rhsOdd = rhsEvenOdd.tail(N);
    solution.head(N) = linop->applyEvenEvenInv(rhsEvenOdd.head(N));
    oddSolution = VectorXcd::Zero(N);
  }

  boost::timer::cpu_timer timer;

  VectorXcd r 
    = (precondition == 1)
    ? rhsOdd - linop->applyPreconditioned(oddSolution)
    : rhs - linop->apply(solution); // 2 * N flops
  double rNorm = r.norm();
  double r0Norm = rNorm;
  int restartFrequency = 20;

  VectorXcd e1 = VectorXcd::Zero(restartFrequency + 1);
  e1(0) = 1.0;

  for (int i = 0; i < maxIterations; ++i) {
    
    MatrixXcd V;
    MatrixXcd H;

    arnoldi(V, H, linop, r, restartFrequency, precondition);
    
    VectorXcd y = H.jacobiSvd(ComputeThinU | ComputeThinV).solve(rNorm * e1);

    if (precondition == 1)
      oddSolution += V.leftCols(restartFrequency) * y;
    else
      solution += V.leftCols(restartFrequency) * y;

    r = (precondition == 1)
      ? rhsOdd - linop->applyPreconditioned(oddSolution)
      : rhs - linop->apply(solution);
    rNorm = r.norm();

    if (rNorm / r0Norm < tolerance) {
      maxIterations = i + 1;
      break;
    }
  }
  tolerance = rNorm / r0Norm;

  boost::timer::cpu_times const elapsedTimes(timer.elapsed());
  boost::timer::nanosecond_type const elapsed(elapsedTimes.system
					      + elapsedTimes.user);

  time = elapsed / 1.0e9;

  if (precondition == 1) {
    solution.tail(N) = oddSolution;
    solution = linop->makeEvenOddSolution(solution);
  }

  return solution;
}
