#include <linear_operators/wilson_hopping_term.hpp>

WilsonHoppingTerm::WilsonHoppingTerm(
  const vector<complex<double> >& boundaryConditions,
  Lattice* lattice) : lattice_(lattice)
{
  // Class constructor - we create a pointer to the lattice and compute the
  // frequently used spin structures used within the Dirac operator.
  this->operatorSize_ 
    = 12 * int(pow(lattice->spatialExtent, 3)) * lattice->temporalExtent;

  for (int i = 0; i < 4; ++i) {
    this->spinStructures_.push_back(Matrix4cd::Identity() - pyQCD::gammas[i]);
  }

  for (int i = 0; i < 4; ++i) {
    this->spinStructures_.push_back(Matrix4cd::Identity() + pyQCD::gammas[i]);
  }

  // Initialise tadpole coefficients
  this->tadpoleCoefficients_[0] = 0.5 / lattice->ut();
  this->tadpoleCoefficients_[1] = 0.5 / lattice->us();
  this->tadpoleCoefficients_[2] = 0.5 / lattice->us();
  this->tadpoleCoefficients_[3] = 0.5 / lattice->us();

  this->nearestNeighbours_ = pyQCD::getNeighbourIndices(1, this->lattice_);
  this->boundaryConditions_ = pyQCD::getBoundaryConditions(1, boundaryConditions,
							   this->lattice_);
}



WilsonHoppingTerm::WilsonHoppingTerm(
  const vector<complex<double> >& boundaryConditions,
  const vector<Matrix4cd>& spinStructures,
  Lattice* lattice) : lattice_(lattice)
{
  // Class constructor - we create a pointer to the lattice and compute the
  // frequently used spin structures used within the Dirac operator.
  this->operatorSize_ 
    = 12 * int(pow(lattice->spatialExtent, 3)) * lattice->temporalExtent;

  for (int i = 0; i < 8; ++i)
    this->spinStructures_.push_back(spinStructures[i]);

  // Initialise tadpole coefficients
  this->tadpoleCoefficients_[0] = 0.5 / lattice->ut();
  this->tadpoleCoefficients_[1] = 0.5 / lattice->us();
  this->tadpoleCoefficients_[2] = 0.5 / lattice->us();
  this->tadpoleCoefficients_[3] = 0.5 / lattice->us();

  this->nearestNeighbours_ = pyQCD::getNeighbourIndices(1, this->lattice_);
  this->boundaryConditions_ = pyQCD::getBoundaryConditions(1, boundaryConditions,
							   this->lattice_);
}



WilsonHoppingTerm::WilsonHoppingTerm(
  const vector<complex<double> >& boundaryConditions,
  const Matrix4cd& spinStructure,
  Lattice* lattice) : lattice_(lattice)
{
  // Class constructor - we create a pointer to the lattice and compute the
  // frequently used spin structures used within the Dirac operator.
  this->operatorSize_ 
    = 12 * int(pow(lattice->spatialExtent, 3)) * lattice->temporalExtent;

  for (int i = 0; i < 8; ++i)
    this->spinStructures_.push_back(spinStructure);

  // Initialise tadpole coefficients
  this->tadpoleCoefficients_[0] = 0.5 / lattice->ut();
  this->tadpoleCoefficients_[1] = 0.5 / lattice->us();
  this->tadpoleCoefficients_[2] = 0.5 / lattice->us();
  this->tadpoleCoefficients_[3] = 0.5 / lattice->us();

  this->nearestNeighbours_ = pyQCD::getNeighbourIndices(1, this->lattice_);
  this->boundaryConditions_ = pyQCD::getBoundaryConditions(1, boundaryConditions,
							   this->lattice_);
}



WilsonHoppingTerm::~WilsonHoppingTerm()
{
  // Empty destructor
}



VectorXcd WilsonHoppingTerm::apply3d(const VectorXcd& psi)
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
	eta(i)
	  -= this->spinStructures_[j](alpha, beta)
	  * this->boundaryConditions_[etaSiteIndex][j]
	  * conj(this->lattice_->getLink(4 * siteBehindIndex + j)(b, a))
	  * psi(12 * siteBehindIndex + 3 * beta + b)
	  * this->tadpoleCoefficients_[j % 4];
	
	eta(i)
	  -= this->spinStructures_[j + 4](alpha, beta)
	  * this->boundaryConditions_[etaSiteIndex][j + 4]
	  * this->lattice_->getLink(4 * etaSiteIndex + j)(a, b)
	  * psi(12 * siteAheadIndex + 3 * beta + b)
	  * this->tadpoleCoefficients_[j % 4];
      }
    }
  }

  return eta;
}



VectorXcd WilsonHoppingTerm::apply(const VectorXcd& psi)
{
  // Right multiply a vector by the operator
  VectorXcd eta(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return VectorXcd::Zero(this->operatorSize_);

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
	asm("#BEGIN TRANSFER");

	complex<double> tempComplexNumbers[] __attribute__((aligned(16)))
	  = {this->spinStructures_[mu](alpha, beta),
	     this->boundaryConditions_[etaSiteIndex][mu],
	     conj(this->lattice_->getLink(4 * siteBehindIndex + mu)(b, a)),
	     psi(12 * siteBehindIndex + 3 * beta + b),
	     this->spinStructures_[mu + 4](alpha, beta),
	     this->boundaryConditions_[etaSiteIndex][mu + 4],
	     this->lattice_->getLink(4 * etaSiteIndex + mu)(a, b),
	     psi(12 * siteAheadIndex + 3 * beta + b)};
	asm("#END TRANSFER");

	asm("#BEGIN FIRST MULT");
	tempComplexNumbers[0] *= tempComplexNumbers[1]
	  * tempComplexNumbers[2] * tempComplexNumbers[3];
	asm("#END FIRST MULT");
	
	asm("#BEGIN SECOND MULT");
	tempComplexNumbers[4] *= tempComplexNumbers[5]
	  * tempComplexNumbers[6] * tempComplexNumbers[7];
	asm("#END SECOND MULT");

	asm("#BEGIN ADDITION");
	tempComplexNumbers[0] += tempComplexNumbers[4];
	asm("#END ADDITION");

	asm("#BEGIN SCALAR MULT");
	tempComplexNumbers[0] *= this->tadpoleCoefficients_[mu % 4];
	asm("#END SCALAR MULT");
	
	asm("#BEGIN ETA SUBTRACT");
	eta(i) = -tempComplexNumbers[0];
	asm("#END ETA SUBTRACT");
      }
    }
  }

  return eta;
}



VectorXcd WilsonHoppingTerm::applyHermitian(const VectorXcd& psi)
{
  VectorXcd eta = this->apply(psi);

  return pyQCD::multiplyGamma5(eta);
}



VectorXcd WilsonHoppingTerm::undoHermiticity(const VectorXcd& psi)
{
  return pyQCD::multiplyGamma5(psi);
}


