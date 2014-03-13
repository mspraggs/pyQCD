#include <linear_operators/unpreconditioned_wilson.hpp>

UnpreconditionedWilson::UnpreconditionedWilson(
    const double mass, const vector<complex<double> >& boundaryConditions,
    Lattice* lattice) : lattice_(lattice)
{
  // Class constructor - we set the fermion mass, create a pointer to the 
  // lattice and compute the frequently used spin structures used within the
  // Dirac operator.
  this->mass_ = mass;
  this->operatorSize_ 
    = 12 * int(pow(lattice->spatialExtent, 3)) * lattice->temporalExtent;

  for (int i = 0; i < 4; ++i) {
    this->spinStructures_.push_back(Matrix4cd::Identity() - pyQCD::gammas[i]);
  }

  for (int i = 0; i < 4; ++i) {
    this->spinStructures_.push_back(Matrix4cd::Identity() + pyQCD::gammas[i]);
  }
  // Here are some Hermitian gamma matrices to use when computing a Hermitian
  // version of the gamma operator
  for (int i = 0; i < 8; ++i) {
    this->hermitianSpinStructures_.push_back(this->spinStructures_[i] 
					     * pyQCD::gamma5);
  }

  int latticeShape[4] = {this->lattice_->temporalExtent,
			 this->lattice_->spatialExtent,
			 this->lattice_->spatialExtent,
			 this->lattice_->spatialExtent};

  for (int i = 0; i < this->operatorSize_ / 12; ++i) {

    vector<int> neighbours(8, 0);
    vector<complex<double> > siteBoundaryConditions(8, complex<double>(1.0,
								       0.0));

    // Determine the coordinates of the site we're on
    int x[5]; // The coordinates of the lattice site (x[4] denotes ignored link)
    pyQCD::getLinkIndices(4 * i, this->lattice_->spatialExtent,
			  this->lattice_->temporalExtent, x);

    int x_minus_mu[5]; // Site/link index for the site/link behind us
    int x_plus_mu[5]; // Site/link index for the site/link in front of us

    // Loop over the four gamma indices (mu) in the sum inside the Wilson action
    for (int mu = 0; mu < 4; ++mu) {

      // Determine whether we need to apply boundary conditions

      copy(x, x + 5, x_minus_mu);
      if (x[mu] - 1 < latticeShape[mu] ||
	  x[mu] - 1 > latticeShape[mu])
	siteBoundaryConditions[mu] = boundaryConditions[mu % 4];

      if (x[mu] + 1 < latticeShape[mu] || x[mu] + 1 > latticeShape[mu])
	siteBoundaryConditions[mu + 4] = boundaryConditions[mu % 4];
          
      // Now we determine the indices of the neighbouring links

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
    this->boundaryConditions_.push_back(siteBoundaryConditions);
    this->nearestNeighbours_.push_back(neighbours);
  }
}



UnpreconditionedWilson::~UnpreconditionedWilson()
{
  // Empty destructor
}



VectorXcd UnpreconditionedWilson::apply(const VectorXcd& psi)
{
  // Right multiply a vector by the operator
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

#pragma omp parallel for
  for (int i = 0; i < this->operatorSize_; ++i) {
    int eta_site_index = i / 12; // Site index of the current row in 
    int alpha = (i % 12) / 3; // Spin index of the current row in eta
    int a = i % 3; // Colour index of the current row in eta

    // Now add the mass term
    eta(i) = (4 + this->mass_) * psi(i);

    // Loop over the four gamma indices (mu) in the sum inside the Wilson action
    for (int mu = 0; mu < 4; ++mu) {
      
      // Now we determine the indices of the neighbouring links

      int x_minus_mu_index = this->nearestNeighbours_[eta_site_index][mu];
      int x_plus_mu_index = this->nearestNeighbours_[eta_site_index][mu + 4];

      // Now loop over spin and colour indices for the portion of the row we're
      // on

      for (int j = 0; j < 12; ++j) {
	int beta = j / 3; // Compute spin
	int b = j % 3; // Compute colour
	eta(i)
	  -= 0.5 * this->spinStructures_[mu](alpha, beta)
	  * this->boundaryConditions_[eta_site_index][mu]
	  * conj(this->lattice_->getLink(4 * x_minus_mu_index + mu)(b, a))
	  * psi(12 * x_minus_mu_index + 3 * beta + b)
	  / this->lattice_->u0();
	
	eta(i)
	  -= 0.5 * this->spinStructures_[mu + 4](alpha, beta)
	  * this->boundaryConditions_[eta_site_index][mu + 4]
	  * this->lattice_->getLink(4 * eta_site_index + mu)(a, b)
	  * psi(12 * x_plus_mu_index + 3 * beta + b)
	  / this->lattice_->u0();
      }
    }
  }

  return eta;
}



VectorXcd UnpreconditionedWilson::applyHermitian(const VectorXcd& psi)
{
  // Right multiply a vector by the operator
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_); // The output vector

  // If psi's the wrong size, get out of here before we segfault
  if (psi.size() != this->operatorSize_)
    return eta;

#pragma omp parallel for
  for (int i = 0; i < this->operatorSize_; ++i) {
    int eta_site_index = i / 12; // Site index of the current row in 
    int alpha = (i % 12) / 3; // Spin index of the current row in eta
    int a = i % 3; // Colour index of the current row in eta

    // Now add the mass term
    eta(i) = (4 + this->mass_) * pyQCD::gamma5(alpha, alpha) * psi(i);

    // Loop over the four gamma indices (mu) in the sum inside the Wilson action
    for (int mu = 0; mu < 4; ++mu) {
      
      // Now we determine the indices of the neighbouring links

      int x_minus_mu_index = this->nearestNeighbours_[eta_site_index][mu];
      int x_plus_mu_index = this->nearestNeighbours_[eta_site_index][mu + 4];

      // Now loop over spin and colour indices for the portion of the row we're
      // on

      for (int j = 0; j < 12; ++j) {
	int beta = j / 3; // Compute spin
	int b = j % 3; // Compute colour
	eta(i)
	  -= 0.5 * this->hermitianSpinStructures_[mu](alpha, beta)
	  * this->boundaryConditions_[eta_site_index][mu]
	  * conj(this->lattice_->getLink(4 * x_minus_mu_index + mu)(b, a))
	  * psi(12 * x_minus_mu_index + 3 * beta + b)
	  / this->lattice_->u0();
	
	eta(i)
	  -= 0.5 * this->hermitianSpinStructures_[mu + 4](alpha, beta)
	  * this->boundaryConditions_[eta_site_index][mu + 4]
	  * this->lattice_->getLink(4 * eta_site_index + mu)(a, b)
	  * psi(12 * x_plus_mu_index + 3 * beta + b)
	  / this->lattice_->u0();
      }
    }
  }

  return eta;
}



VectorXcd UnpreconditionedWilson::undoHermiticity(const VectorXcd& psi)
{
  VectorXcd eta = VectorXcd::Zero(this->operatorSize_);

#pragma omp parallel for
  for (int i = 0; i < this->operatorSize_ / 12; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 4; ++k)
	for (int l = 0; l < 3; ++l)
	  eta(12 * i + 3 * j + l)
	    += pyQCD::gamma5(j, k) * psi(12 * i + 3 * k + l);

  return eta;
}


