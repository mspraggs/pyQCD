#include <lattice.hpp>
#include <utils.hpp>

void Lattice::print()
{
  // Print the links out. A bit redundant due to the interfaces library,
  // but here in case it's needed.
  for (int i = 0; i < this->nLinks_; ++i) {
    cout << this->links_[i] << endl;
  }
}



Matrix3cd& Lattice::getLink(const int link[5])
{
  // Return link specified by index (sanitizes link indices)
  int tempLink[5];
  tempLink[0] = pyQCD::mod(link[0], this->temporalExtent);
  for (int i = 1; i < 4; ++i)
    tempLink[i] = pyQCD::mod(link[i], this->spatialExtent);
  tempLink[4] = pyQCD::mod(link[4], 4);

  int index = pyQCD::getLinkIndex(tempLink, this->spatialExtent);
  
  return this->links_[index];
}



Matrix3cd& Lattice::getLink(const vector<int> link)
{
  // Return link specified by indices
  int tempLink[5];
  tempLink[0] = pyQCD::mod(link[0], this->temporalExtent);
  for (int i = 1; i < 4; ++i)
    tempLink[i] = pyQCD::mod(link[i], this->spatialExtent);
  tempLink[4] = pyQCD::mod(link[4], 4);

  int index = pyQCD::getLinkIndex(tempLink, this->spatialExtent);

  return this->links_[index];
}



void Lattice::setLink(const int link[5], const Matrix3cd& matrix)
{
  // Set the value of a link
  this->getLink(link) = matrix;
}



GaugeField Lattice::getSubLattice(const int startIndex, const int size)
{
  // Returns a GaugeField object corresponding to the sub-lattice starting at
  // link index startIndex

  GaugeField out;
  out.resize(size * size * size * size * 4);

  int incrementOne = 4 * this->spatialExtent;
  int incrementTwo = incrementOne * this->spatialExtent;
  int incrementThree = incrementTwo * this->spatialExtent;
  
  int index = 0;

  for (int i = 0; i < size * incrementThree; i += incrementThree) {
    for (int j = 0; j < size * incrementTwo; j += incrementTwo) {
      for (int k = 0; k < size * incrementOne; k += incrementOne) {
	for (int l = 0; l < 4 * size; ++l) {
	  out[index] = this->links_[i + j + k + l];
	  ++index;
	}
      }
    }
  }

  return out;
}



Matrix2cd Lattice::makeHeatbathSu2(double coefficients[4],
				   const double weighting)
{
  // Generate a random SU2 matrix distributed according to heatbath
  // (See Gattringer and Lang)
  // Initialise lambdaSquared so that we'll go into the for loop
  double lambdaSquared = 2.0;
  // A random squared float to use in the while loop
  double randomSquare = pow(rng.generateReal(), 2);
  // Loop until lambdaSquared meets the distribution condition
  while (randomSquare > 1.0 - lambdaSquared) {
    // Generate three random floats in (0,1] as per Gattringer and Lang
    double r1 = 1 - rng.generateReal();
    double r2 = 1 - rng.generateReal();
    double r3 = 1 - rng.generateReal();
    // Need a factor of 1.5 here rather that 1/3, not sure why...
    // Possibly due to Nc = 3 in this case
    lambdaSquared = - 1.5 / (weighting * this->beta_) *
      (log(r1) + pow(cos(2 * pyQCD::pi * r2), 2) * log(r3));

    // Get a new random number
    randomSquare = pow(rng.generateReal(), 2);
  }

  // Get the first of the four elements needed to specify the SU(2)
  // matrix using Pauli matrices
  coefficients[0] = 1 - 2 * lambdaSquared;
  // Magnitude of remaing three-vector is given as follows
  double xMag = sqrt(abs(1 - coefficients[0] * coefficients[0]));

  // Randomize the direction of the remaining three-vector
  // Get a random cos(theta) in [0,1)
  double costheta = -1.0 + 2.0 * rng.generateReal();
  // And a random phi in [0,2*pi)
  double phi = 2 * pyQCD::pi * rng.generateReal();

  // We now have everything we need to calculate the remaining three
  // components, so do it
  coefficients[1] = xMag * sqrt(1 - costheta * costheta) * cos(phi);
  coefficients[2] = xMag * sqrt(1 - costheta * costheta) * sin(phi);
  coefficients[3] = xMag * costheta;

  // Now get the SU(2) matrix
  return pyQCD::createSu2(coefficients);
}



Matrix3cd Lattice::makeRandomSu3()
{
  // Generate a random SU3 matrix, weighted by epsilon
  Matrix3cd A;
  // First generate a random matrix whos elements all lie in/on unit circle  
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      A(i, j) = rng.generateReal();
      A(i, j) *= exp(2  * pyQCD::pi * pyQCD::i * rng.generateReal());
    }
  }
  // Weight the matrix with weighting eps
  A *= 0.24;
  // Make the matrix traceless and Hermitian
  A(2, 2) = -(A(1, 1) + A(0, 0));
  Matrix3cd B = 0.5 * (A - A.adjoint());
  return B.exp();
}



void Lattice::reunitarize()
{
  // Do fast polar decomp on all gauge field matrices to ensure they're unitary
  if (this->parallelFlag_ == 1) {
#pragma omp parallel for
    for (int i = 0; i < this->nLinks_; ++i) {
      double check = 1.0;
      while (check > 1e-15) {
	Matrix3cd oldMatrix = this->links_[i];
	Matrix3cd inverseMatrix = this->links_[i].inverse();
	double gamma = sqrt(pyQCD::oneNorm(inverseMatrix)
			    / pyQCD::oneNorm(this->links_[i]));
	this->links_[i] *= 0.5 * gamma;
	this->links_[i] += 0.5 / gamma * inverseMatrix.adjoint();
	oldMatrix -= this->links_[i];
	check = sqrt(pyQCD::oneNorm(oldMatrix));
      }
    }
  }
  else {
    for (int i = 0; i < this->nLinks_; ++i) {
      double check = 1.0;
      while (check > 1e-15) {
	Matrix3cd oldMatrix = this->links_[i];
	Matrix3cd inverseMatrix = this->links_[i].inverse();
	double gamma = sqrt(pyQCD::oneNorm(inverseMatrix)
			    / pyQCD::oneNorm(this->links_[i]));
	this->links_[i] *= 0.5 * gamma;
	this->links_[i] += 0.5 / gamma * inverseMatrix.adjoint();
	oldMatrix -= this->links_[i];
	check = sqrt(pyQCD::oneNorm(oldMatrix));
      }
    }
  }
}
