#include <linear_operators/jacobi_smearing.hpp>

JacobiSmearing::JacobiSmearing(
    const int numSmears, const double smearingParameter,
    const vector<complex<double> >& boundaryConditions,
    Lattice* lattice) : lattice_(lattice)
{
  // Class constructor - we set the pointer to the lattice, determine
  // the nearest neighbours and create the identity for our spin
  // structure

  this->operatorSize_ 
    = 12 * int(pow(lattice->spatialExtent, 3)) * lattice->temporalExtent;

  this->identity_ = Matrix4cd::Identity();
  this->smearingParameter_ = smearingParameter;
  this->numSmears_ = numSmears;

  // Initialise tadpole coefficients
  this->tadpoleCoefficients_[0] = lattice->ut();
  this->tadpoleCoefficients_[1] = lattice->us();
  this->tadpoleCoefficients_[2] = lattice->us();
  this->tadpoleCoefficients_[3] = lattice->us();

  int latticeShape[4] = {this->lattice_->temporalExtent,
			 this->lattice_->spatialExtent,
			 this->lattice_->spatialExtent,
			 this->lattice_->spatialExtent};

  for (int i = 0; i < this->operatorSize_ / 12; ++i) {

    vector<int> neighbours(8, 0);
    vector<complex<double> > siteBoundaryConditions(8, complex<double>(1.0,
								       0.0));

    // Determine the coordinates of the site we're on
    int site[5]; // The coordinates of the lattice site/link
    pyQCD::getLinkIndices(4 * i, this->lattice_->spatialExtent,
			  this->lattice_->temporalExtent, site);

    int siteBehind[5]; // Site/link index for the site/link behind us
    int siteAhead[5]; // Site/link index for the site/link in front of us

    // Loop over the three gamma indices (j) in the sum inside the Wilson action
    for (int j = 0; j < 4; ++j) {
          
      // Now we determine the indices of the neighbouring links

      copy(site, site + 5, siteBehind);
      if (site[j] - 1 < 0 || site[j] - 1 >= latticeShape[j])
	siteBoundaryConditions[j] = boundaryConditions[j % 4];

      if (site[j] + 1 < 0 || site[j] + 1 >= latticeShape[j])
	siteBoundaryConditions[j + 4] = boundaryConditions[j % 4];

      siteBehind[j] = pyQCD::mod(siteBehind[j] - 1, latticeShape[j]);
      int siteBehindIndex = pyQCD::getLinkIndex(siteBehind,
						 latticeShape[1]) / 4;

      copy(site, site + 5, siteAhead);
      siteAhead[j] = pyQCD::mod(siteAhead[j] + 1, latticeShape[j]);
      int siteAheadIndex = pyQCD::getLinkIndex(siteAhead,
						latticeShape[1]) / 4;

      neighbours[j] = siteBehindIndex;
      neighbours[j + 4] = siteAheadIndex;
    }
    this->boundaryConditions_.push_back(siteBoundaryConditions);
    this->nearestNeighbours_.push_back(neighbours);
  }
}



JacobiSmearing::~JacobiSmearing()
{
  // Empty destructor
}



VectorXcd JacobiSmearing::apply(const VectorXcd& psi)
{
  // Apply the smearing operator itself

  VectorXcd eta = psi; // The output

  // Temporary vector to use in the sum
  VectorXcd tempTerm = VectorXcd::Zero(psi.size());

  for (int i = 0; i < this->numSmears_; ++i) {
    tempTerm = this->smearingParameter_ * this->applyOnce(tempTerm);
    eta += tempTerm;
  }

  return eta;
}



VectorXcd JacobiSmearing::applyOnce(const VectorXcd& psi)
{
  // Right multiply a vector once by the operator H (see Jacobi smearing in
  // Gattringer and Lang).
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

#pragma omp parallel for
  for (int i = 0; i < this->operatorSize_; ++i) {
    int etaSiteIndex = i / 12; // Site index of the current row in eta
    int alpha = (i % 12) / 3; // Spin index of the current row in eta
    int a = i % 3; // Colour index of the current row in eta

    // Loop over the three spatial directions in the sum inside the
    // smearing operator
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
	  -= 0.5 * this->identity_(alpha, beta)
	  * this->boundaryConditions_[etaSiteIndex][j]
	  * conj(this->lattice_->getLink(4 * siteBehindIndex + j)(b, a))
	  * psi(12 * siteBehindIndex + 3 * beta + b)
	  / this->tadpoleCoefficients_[i % 4];
	
	eta(i)
	  -= 0.5 * this->identity_(alpha, beta)
	  * this->boundaryConditions_[etaSiteIndex][j + 4]
	  * this->lattice_->getLink(4 * etaSiteIndex + j)(a, b)
	  * psi(12 * siteAheadIndex + 3 * beta + b)
	  / this->tadpoleCoefficients_[i % 4];
      }
    }
  }

  return eta;
}



VectorXcd JacobiSmearing::applyHermitian(const VectorXcd& psi)
{
  // Right multiply a vector by the operator
  return this->apply(psi);
}



VectorXcd JacobiSmearing::undoHermiticity(const VectorXcd& psi)
{
  // Undo Hermticity in applyHermtian (there is none, so retur psi)
  return psi;
}
