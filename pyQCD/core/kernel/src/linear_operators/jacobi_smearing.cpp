#include <linear_operators/jacobi_smearing.hpp>

JacobiSmearing::JacobiSmearing(const int numSmears,
			       const double smearingParameter,
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

  int latticeShape[4] = {this->lattice_->temporalExtent,
			 this->lattice_->spatialExtent,
			 this->lattice_->spatialExtent,
			 this->lattice_->spatialExtent};

  for (int i = 0; i < this->operatorSize_ / 12; ++i) {

    vector<int> neighbours(8, 0);

    // Determine the coordinates of the site we're on
    int x[5]; // The coordinates of the lattice site (x[4] denotes ignored link)
    pyQCD::getLinkIndices(4 * i, this->lattice_->spatialExtent,
			  this->lattice_->temporalExtent, x);

    int x_minus_mu[5]; // Site/link index for the site/link behind us
    int x_plus_mu[5]; // Site/link index for the site/link in front of us

    // Loop over the four gamma indices (mu) in the sum inside the Wilson action
    for (int mu = 0; mu < 4; ++mu) {
          
      // Now we determine the indices of the neighbouring links

      copy(x, x + 5, x_minus_mu);
      x_minus_mu[mu] = pyQCD::mod(x_minus_mu[mu] - 1, latticeShape[mu]);
      int x_minus_mu_index = pyQCD::getLinkIndex(x_minus_mu,
						 latticeShape[1]) / 4;
      // Set the link direction we're currently on as we'll need this later
      x_minus_mu[4] = mu;

      copy(x, x + 5, x_plus_mu);
      x_plus_mu[mu] = pyQCD::mod(x_plus_mu[mu] + 1, latticeShape[mu]);
      int x_plus_mu_index = pyQCD::getLinkIndex(x_plus_mu,
						latticeShape[1]) / 4;

      neighbours[mu] = x_minus_mu_index;
      neighbours[mu + 4] = x_plus_mu_index;
    }
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

  VectorXcd eta = VectorXcd::Zero(psi.size()); // The output

  // Temporary vector to use in the sum
  VectorXcd tempTerm = psi;

  for (int i = 0; i < this->numSmears_; ++i) {
    eta += tempTerm;

    tempTerm = this->smearingParameter_ * this->applyOnce(tempTerm);
  }

  return eta;
}



VectorXcd JacobiSmearing::applyOnce(const VectorXcd& psi)
{
  // Right multiply a vector once by the operator
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

#pragma omp parallel for
  for (int i = 0; i < this->operatorSize_; ++i) {
    int eta_site_index = i / 12; // Site index of the current row in 
    int alpha = (i % 12) / 3; // Spin index of the current row in eta
    int a = i % 3; // Colour index of the current row in eta

    // Loop over the three spatial directions in the sum inside the
    // smearing operator
    for (int j = 1; j < 4; ++j) {
      
      // Now we determine the indices of the neighbouring links

      int x_minus_j_index = this->nearestNeighbours_[eta_site_index][j];
      int x_plus_j_index = this->nearestNeighbours_[eta_site_index][j + 4];

      // Now loop over spin and colour indices for the portion of the row we're
      // on

      for (int j = 0; j < 12; ++j) {
	int beta = j / 3; // Compute spin
	int b = j % 3; // Compute colour
	eta(i)
	  -= 0.5 * this->identity_(alpha, beta)
	  * conj(this->lattice_->getLink(4 * x_minus_j_index + j)(b, a))
	  * psi(12 * x_minus_j_index + 3 * beta + b)
	  / this->lattice_->u0();
	
	eta(i)
	  -= 0.5 * this->identity_(alpha, beta)
	  * this->lattice_->getLink(4 * eta_site_index + j)(a, b)
	  * psi(12 * x_plus_j_index + 3 * beta + b)
	  / this->lattice_->u0();
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
