#include <linear_operators/hopping_term.hpp>

HoppingTerm::HoppingTerm(
  const vector<complex<double> >& boundaryConditions,
  Lattice* lattice, const int nHops) : LinearOperator::LinearOperator()
{
  // Class constructor - we create a pointer to the lattice and compute the
  // frequently used spin structures used within the Dirac operator.

  for (int i = 0; i < 4; ++i) {
    this->spinStructures_.push_back(Matrix4cd::Identity() - pyQCD::gammas[i]);
  }

  for (int i = 0; i < 4; ++i) {
    this->spinStructures_.push_back(Matrix4cd::Identity() + pyQCD::gammas[i]);
  }

  this->init(lattice, boundaryConditions, nHops);
}



HoppingTerm::HoppingTerm(
  const vector<complex<double> >& boundaryConditions,
  const vector<Matrix4cd>& spinStructures,
  Lattice* lattice, const int nHops) : lattice_(lattice)
{
  // Class constructor - we create a pointer to the lattice and compute the
  // frequently used spin structures used within the Dirac operator.

  for (int i = 0; i < 8; ++i)
    this->spinStructures_.push_back(spinStructures[i]);

  this->init(lattice, boundaryConditions, nHops);
}



HoppingTerm::HoppingTerm(
  const vector<complex<double> >& boundaryConditions,
  const Matrix4cd& spinStructure,
  Lattice* lattice, const int nHops) : lattice_(lattice)
{
  // Class constructor - we create a pointer to the lattice and compute the
  // frequently used spin structures used within the Dirac operator.

  for (int i = 0; i < 8; ++i)
    this->spinStructures_.push_back(spinStructure);

  this->init(lattice, boundaryConditions, nHops);
}



HoppingTerm::~HoppingTerm()
{
  // Empty destructor
}



void HoppingTerm::init(Lattice* lattice,
		       const vector<complex<double> >& boundaryConditions,
		       const int nHops)
{
  // Code common to all constructors
  this->operatorSize_ 
    = 12 * int(pow(lattice->spatialExtent, 3)) * lattice->temporalExtent;
  this->lattice_ = lattice;

  // Initialise tadpole coefficients
  this->tadpoleCoefficients_[0] = 1.0 / pow(lattice->ut(), nHops);
  this->tadpoleCoefficients_[1] = 1.0 / pow(lattice->us(), nHops);
  this->tadpoleCoefficients_[2] = 1.0 / pow(lattice->us(), nHops);
  this->tadpoleCoefficients_[3] = 1.0 / pow(lattice->us(), nHops);

  this->nearestNeighbours_ = pyQCD::getNeighbourIndices(nHops, this->lattice_);
  this->boundaryConditions_ = pyQCD::getBoundaryConditions(nHops,
							   boundaryConditions,
							   this->lattice_);

  // Now set up the links for hopping
  this->links_.resize(lattice->nLinks());

  for (int i = 0; i < this->lattice_->temporalExtent; ++i) {
    for (int j = 0; j < this->lattice_->spatialExtent; ++j) {      
      for (int k = 0; k < this->lattice_->spatialExtent; ++k) {	
	for (int l = 0; l < this->lattice_->spatialExtent; ++l) {
	  for (int m = 0; m < 4; ++m) {
	    int link[5] = {i, j, k, l, m};
	    int hoppingLink[5] = {i, j, k, l, m};
	    this->links_[pyQCD::getLinkIndex(link,
					     this->lattice_->spatialExtent)]
	      = Matrix3cd::Identity();

	    for (int n = 0; n < nHops; ++n) {
	      hoppingLink[m]++;
	      this->links_[pyQCD::getLinkIndex(link,
					       this->lattice_->spatialExtent)]
		*= this->lattice_->getLink(hoppingLink);
	    }
	  }
	}
      }
    }
  }
}



VectorXcd HoppingTerm::apply3d(const VectorXcd& psi)
{
  // This is the 3d version used for Jacobi smearing and so on
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

#pragma omp parallel for
  for (int i = 0; i < this->operatorSize_; ++i) {
    int etaSiteIndex = i / 12; // Site index of the current row in 
    int alpha = (i % 12) / 3; // Spin index of the current row in eta
    int a = i % 3; // Colour index of the current row in eta

    // Loop over the three gamma indices (j) in the sum inside the Wilson action
    for (int j = 1; j < 4; ++j) {
      
      // Now we determine the indices of the neighbouring links

      int siteBehindIndex = this->nearestNeighbours_[etaSiteIndex][j];
      int siteAheadIndex = this->nearestNeighbours_[etaSiteIndex][j + 4];

      // Now loop over spin and colour indices for the portion of the row we're
      // on

      for (int k = 0; k < 12; ++k) {
	int beta = k / 3; // Compute spin
	int b = k % 3; // Compute colour

	complex<double> tempComplexNumbers[] __attribute__((aligned(16)))
	  = {this->spinStructures_[j](alpha, beta),
	     this->boundaryConditions_[etaSiteIndex][j],
	     conj(this->links_[4 * siteBehindIndex + j](b, a)),
	     psi(12 * siteBehindIndex + k),
	     this->spinStructures_[j + 4](alpha, beta),
	     this->boundaryConditions_[etaSiteIndex][j + 4],
	     this->links_[4 * etaSiteIndex + j](a, b),
	     psi(12 * siteAheadIndex + k)};
	// 4 * 6 = 24 flops
	tempComplexNumbers[0] *= tempComplexNumbers[1]
	  * tempComplexNumbers[2] * tempComplexNumbers[3];
	// 4 * 6 = 24 flops
	tempComplexNumbers[4] *= tempComplexNumbers[5]
	  * tempComplexNumbers[6] * tempComplexNumbers[7];
	
	tempComplexNumbers[0] += tempComplexNumbers[4]; // 2 flops
	tempComplexNumbers[0] *= this->tadpoleCoefficients_[j]; // 2 flops
	
	eta(i) += tempComplexNumbers[0]; // 2 flops

	// Total flops inside this loop = 2 * 24 + 2 + 2 + 2 = 54 flops
      }
    }
  }

  this->nFlops_ += this->operatorSize_ * 3 * 12 * 54;

  return eta;
}



VectorXcd HoppingTerm::apply(const VectorXcd& psi)
{
  // Right multiply a vector by the operator
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

#pragma omp parallel for
  for (int i = 0; i < this->operatorSize_; ++i) {
    int etaSiteIndex = i / 12; // Site index of the current row in 
    int alpha = (i % 12) / 3; // Spin index of the current row in eta
    int a = i % 3; // Colour index of the current row in eta

    // Loop over the four gamma indices (mu) in the sum inside the Wilson action
    for (int mu = 0; mu < 4; ++mu) {
      
      // Now we determine the indices of the neighbouring links

      int siteBehindIndex = this->nearestNeighbours_[etaSiteIndex][mu];
      int siteAheadIndex = this->nearestNeighbours_[etaSiteIndex][mu + 4];

      // Now loop over spin and colour indices for the portion of the row we're
      // on

      for (int j = 0; j < 12; ++j) {
	int beta = j / 3; // Compute spin
	int b = j % 3; // Compute colour

	complex<double> tempComplexNumbers[] __attribute__((aligned(16)))
	  = {this->spinStructures_[mu](alpha, beta),
	     this->boundaryConditions_[etaSiteIndex][mu],
	     conj(this->links_[4 * siteBehindIndex + mu](b, a)),
	     psi(12 * siteBehindIndex + j),
	     this->spinStructures_[mu + 4](alpha, beta),
	     this->boundaryConditions_[etaSiteIndex][mu + 4],
	     this->links_[4 * etaSiteIndex + mu](a, b),
	     psi(12 * siteAheadIndex + j)};
	
	tempComplexNumbers[0] *= tempComplexNumbers[1]
	  * tempComplexNumbers[2] * tempComplexNumbers[3];
	
	tempComplexNumbers[4] *= tempComplexNumbers[5]
	  * tempComplexNumbers[6] * tempComplexNumbers[7];

	tempComplexNumbers[0] += tempComplexNumbers[4];
	tempComplexNumbers[0] *= this->tadpoleCoefficients_[mu];
	
	eta(i) += tempComplexNumbers[0];
      }
    }
  }

  this->nFlops_ += this->operatorSize_ * 4 * 12 * 54;

  return eta;
}



VectorXcd HoppingTerm::applyHermitian(const VectorXcd& psi)
{
  VectorXcd eta = this->apply(psi);

  this->nFlops_ += 4 * this->operatorSize_ * 8;

  return pyQCD::multiplyGamma5(eta);
}



VectorXcd HoppingTerm::makeHermitian(const VectorXcd& psi)
{
  return pyQCD::multiplyGamma5(psi);
}


