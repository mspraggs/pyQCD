#include <lattice.hpp>
#include <utils.hpp>

SparseMatrix<complex<double> > Lattice::computeDiracMatrix(const double mass,
							   const double spacing)
{
  // Calculates the Dirac matrix for the current field configuration
  // using Wilson fermions
  
  // Create the sparse matrix we're going to return
  SparseMatrix<complex<double> > out(3 * this->nLinks_, 3 * this->nLinks_);

  vector<Tlet> tripletList;
  tripletList.reserve((1 + 12 * 8) * 3 * this->nLinks_);
  for (int i = 0; i < 3 * this->nLinks_; ++i) {
    tripletList.push_back(Tlet(i, i, mass + 4 / spacing));
  }

  vector<vector<Tlet> > tempTripletList;
  tempTripletList.resize(this->nLinks_ / 4);
  
  // Now iterate through the matrix and add the neighbouring elements
#pragma omp parallel for
  for (int i = 0; i < this->nLinks_ / 4; ++i) {
    int rowLink[5];
    pyQCD::getLinkIndices(4 * i, this->spatialExtent, this->temporalExtent,
			  rowLink);

    vector<Tlet> subTempTripletList;

    // We've already calculated the eight neighbours, so we'll deal with those
    // alone
    for (int j = 0; j < 8; ++j) {
      // Get the dimension and index of the current neighbour
      int columnIndex = this->propagatorColumns_[i][j][0];
      int dimension = (this->propagatorColumns_[i][j][1] > 3)
	? this->propagatorColumns_[i][j][1] - 4
	: this->propagatorColumns_[i][j][1];

      int boundaryCondition
	= (dimension == 0) ? this->propagatorColumns_[i][j][2] : 1;

      // Now we'll get the relevant colour and spin matrices
      Matrix3cd colourMatrix;
      Matrix4cd spinMatrix;
      // (See the action for what's going on here.)
      if (this->propagatorColumns_[i][j][1] > 3) {
	rowLink[4] = dimension;
	colourMatrix = boundaryCondition * this->getLink(rowLink);
	spinMatrix = Matrix4cd::Identity() + pyQCD::gammas[dimension];
      }
      else {
	rowLink[dimension]--;
	rowLink[4] = dimension;
	colourMatrix = boundaryCondition * this->getLink(rowLink).adjoint();
	rowLink[dimension]++;
	spinMatrix = Matrix4cd::Identity() - pyQCD::gammas[dimension];
      }
      // Now loop through the two matrices and put the non-zero entries in the
      // triplet list (basically a tensor product, could put this in a utility
      // function as this might be something that needs to be done again if
      // other actions are implemented).

      for (int k = 0; k < 4; ++k) {
	for (int m = 0; m < 3; ++m) {
	  for (int l = 0; l < 4; ++l) {
	    for (int n = 0; n < 3; ++n) {
	      complex<double> sum = -0.5 / spacing
		* spinMatrix(k, l) * colourMatrix(m, n);
	      if (sum != complex<double>(0,0))
		subTempTripletList.push_back(Tlet(12 * i + 3 * k + m,
						  3 * columnIndex + 3 * l + n,
						  sum));
	    }
	  }
	}
      }
    }
    tempTripletList[i].resize(subTempTripletList.size());
    tempTripletList[i] = subTempTripletList;
  }

  for (int i = 0; i < this->nLinks_ / 4; ++i)
    for (unsigned int j = 0; j < tempTripletList[i].size(); ++j)
      tripletList.push_back(tempTripletList[i][j]);
  
  // Add all the triplets to the sparse matrix
  out.setFromTriplets(tripletList.begin(), tripletList.end());
  
  return out;
}



SparseMatrix<complex<double> >
Lattice::computeSmearingOperator(const double smearingParameter,
				 const int nSmears)
{
  // Create the sparse matrix we're going to return
  SparseMatrix<complex<double> > out(3 * this->nLinks_, 3 * this->nLinks_);
  out.setZero();
  // Create the sparse matrix H (eqn 6.40 of Gattringer and Lang)
  SparseMatrix<complex<double> > matrixH(3 * this->nLinks_, 3 * this->nLinks_);

  // Only bother setting up matrixH if there's smearing to do
  if (nSmears > 0) {

    // This is where we'll store the matrix entries before intialising the matrix
    vector<Tlet> tripletList;
    tripletList.reserve(12 * 8 * 3 * this->nLinks_);

    vector<vector<Tlet> > tempTripletList;
    tempTripletList.resize(this->nLinks_ / 4);
    
    // Now iterate through the matrix and add the neighbouring elements
#pragma omp parallel for
    for (int i = 0; i < this->nLinks_ / 4; ++i) {
      int rowLink[5];
      pyQCD::getLinkIndices(4 * i, this->spatialExtent, this->temporalExtent,
			    rowLink);

      vector<Tlet> subTempTripletList;
      
      // We've already calculated the eight neighbours, so we'll deal with those
      // alone
      // And we're not interested in the temporal links
      int indices[6] = {1, 2, 3, 5, 6, 7};
      for (int j = 0; j < 6; ++j) {
	// Get the dimension and index of the current neighbour
	int columnIndex = this->propagatorColumns_[i][indices[j]][0];
	
	// Now we'll get the relevant colour and spin matrices
	Matrix3cd colourMatrix;
	Matrix4cd spinMatrix = Matrix4cd::Identity();
	// (See the action for what's going on here.)
	if (this->propagatorColumns_[i][indices[j]][1] > 3) {
	  int dimension = this->propagatorColumns_[i][indices[j]][1] - 4;
	  rowLink[4] = dimension;
	  colourMatrix = this->getLink(rowLink);
	}
	else {
	  int dimension = this->propagatorColumns_[i][indices[j]][1];
	  rowLink[dimension]--;
	  rowLink[4] = dimension;
	  colourMatrix = this->getLink(rowLink).adjoint();
	  rowLink[dimension]++;
	}
	// Now loop through the two matrices and put the non-zero entries in the
	// triplet list (basically a tensor product, could put this in a utility
	// function as this might be something that needs to be done again if
	// other actions are implemented).
	for (int k = 0; k < 4; ++k) {
	  for (int m = 0; m < 3; ++m) {
	    for (int l = 0; l < 4; ++l) {
	      for (int n = 0; n < 3; ++n) {
		complex<double> sum = spinMatrix(k, l) * colourMatrix(m, n);
#pragma omp critical
		if (sum != complex<double>(0,0))
		  subTempTripletList.push_back(Tlet(12 * i + 3 * k + m,
						    3 * columnIndex + 3 * l + n,
						    sum));
	      }
	    }
	  }
	}
      }
    tempTripletList[i].resize(subTempTripletList.size());
    tempTripletList[i] = subTempTripletList;
    }

    for (int i = 0; i < this->nLinks_ / 4; ++i)
      for (unsigned int j = 0; j < tempTripletList[i].size(); ++j)
	tripletList.push_back(tempTripletList[i][j]);
    
    // Add all the triplets to the sparse matrix
    matrixH.setFromTriplets(tripletList.begin(), tripletList.end());
    
  }
  
  // Need a sparse
  vector<Tlet> identityTripletList;
  for (int i = 0; i < 3 * this->nLinks_; ++i) {
    identityTripletList.push_back(Tlet(i, i, complex<double>(1.0, 0.0)));
  }
  
  // Create an identity matrix that'll hold the matrix powers in the sum below
  SparseMatrix<complex<double> > matrixHPower(3 * this->nLinks_,
					      3 * this->nLinks_);
  matrixHPower.setFromTriplets(identityTripletList.begin(),
			       identityTripletList.end());
  
  // Now do the sum as in eqn 6.40 of G+L
  for (int i = 0; i <= nSmears; ++i) {
    out += pow(smearingParameter, i) * matrixHPower;
    // Need to do the matrix power by hand
    matrixHPower = matrixHPower * matrixH;
  }
  
  return out;
}



VectorXcd
Lattice::makeSource(const int site[4], const int spin, const int colour,
		    const SparseMatrix<complex<double> >& smearingOperator)
{
  // Generate a (possibly smeared) quark source at the given site, spin and
  // colour
  int nIndices = 3 * this->nLinks_;
  VectorXcd source(nIndices);
  source.setZero(nIndices);

  // Index for the vector point source
  int spatial_index = pyQCD::getLinkIndex(site[0], site[1], site[2], site[3], 0,
					  this->spatialExtent);
	
  // Set the point source
  int index = colour + 3 * (spin + spatial_index);
  source(index) = 1.0;

  // Now apply the smearing operator
  source = smearingOperator * source;

  return source;
}



vector<MatrixXcd>
Lattice::computePropagator(const double mass, const double spacing, int site[4],
			   const SparseMatrix<complex<double> >& D,
			   const int nSourceSmears,
			   const double sourceSmearingParameter,
			   const int nSinkSmears,
			   const double sinkSmearingParameter,
			   const int solverMethod,
			   const int verbosity)
{
  // Computes the propagator vectors for the 12 spin-colour indices at
  // the given lattice site, using the Dirac operator

  // How many indices are we dealing with?
  int nSites = this->nLinks_ / 4;

#ifdef USE_CUDA
  // Index for the vector point source
  int spatialIndex = pyQCD::getLinkIndex(site[0], site[1], site[2], site[3], 0,
					 this->spatialExtent);
#endif
  // Declare a variable to hold our propagator
  vector<MatrixXcd> propagator(nSites, MatrixXcd::Zero(12, 12));

  // Compute the source and sink smearing operators
  SparseMatrix<complex<double> > sourceSmearingOperator
    = computeSmearingOperator(sourceSmearingParameter, nSourceSmears);
  SparseMatrix<complex<double> > sinkSmearingOperator
    = computeSmearingOperator(sinkSmearingParameter, nSinkSmears);
  
  vector<complex<double> > boundaryConditions(4, complex<double>(1.0, 0.0));
  boundaryConditions[0] = complex<double>(-1.0, 0.0);
  LinearOperator* linop = new UnpreconditionedWilson(mass, boundaryConditions,
						     this);

  // If using CG, then we need to multiply D by its adjoint
  if (solverMethod == 1) {
    // Get adjoint matrix 
    SparseMatrix<complex<double> > Dadj = D.adjoint();

    // The matrix we'll be inverting
    SparseMatrix<complex<double> > M = D * Dadj;
#ifdef USE_CUDA
    pyQCD::cudaCG(M, Dadj, sourceSmearingOperator, sinkSmearingOperator,
		  spatialIndex, propagator, verbosity);
#else
    // And the solver
    //ConjugateGradient<SparseMatrix<complex<double> > > solver(M);
    //solver.setMaxIterations(1000);
    //solver.setTolerance(1e-8);

    // Loop through colour and spin indices and invert propagator
    for (int i = 0; i < 4; ++i) {
      for(int j = 0; j < 3; ++j) {
	if (verbosity > 0)
	  cout << "  Inverting for spin " << i
	       << " and colour " << j << "..." << endl;
	// Create the source vector
	VectorXcd source = this->makeSource(site, i, j, sourceSmearingOperator);
	
	// Do the inversion
	double residual = 0.0;
	int iterations = 0;
	VectorXcd solution = cg(linop, source, 1e-4, 1000, residual, iterations);

	// Smear the sink
	solution = sinkSmearingOperator * solution;
	
	// Add the result to the propagator matrix
	for (int k = 0; k < nSites; ++k) {
	  for (int l = 0; l < 12; ++l) {
	    propagator[k](l, j + 3 * i) = solution(12 * k + l);
	  }
	}
	if (verbosity > 0)
	  cout << "  -> Inversion reached tolerance of "
	       << residual << " in " << iterations
	       << " iterations." << endl;
      }
    }
#endif
  }
  else {
#ifdef USE_CUDA
    pyQCD::cudaBiCGstab(D, sourceSmearingOperator, sinkSmearingOperator,
			spatialIndex, propagator, verbosity);
#else
    // Otherwise just use BiCGSTAB
    //BiCGSTAB<SparseMatrix<complex<double> > > solver(D);
    //solver.setMaxIterations(1000);
    //solver.setTolerance(1e-8);
    
    // Loop through colour and spin indices and invert propagator
    for (int i = 0; i < 4; ++i) {
      for(int j = 0; j < 3; ++j) {
	if (verbosity > 0)
	  cout << "  Inverting for spin " << i
	       << " and colour " << j << "..." << endl;
	// Create the source vector
	VectorXcd source = this->makeSource(site, i, j, sourceSmearingOperator);
	
	// Do the inversion
	double residual = 0.0;
	int iterations = 0;
	VectorXcd solution = bicgstab(linop, source, 1e-4, 1000, residual,
				      iterations);

	// Smear the sink
	solution = sinkSmearingOperator * solution;
	
	// Add the result to the propagator matrix
	for (int k = 0; k < nSites; ++k) {
	  for (int l = 0; l < 12; ++l) {
	    propagator[k](l, j + 3 * i) = solution(12 * k + l);
	  }
	}
	if (verbosity > 0)
	  cout << "  -> Inversion reached tolerance of "
	       << residual << " in " << iterations
	       << " iterations." << endl;
      }
    }
#endif
  }

  delete linop;

  return propagator;
}



vector<MatrixXcd>
Lattice::computePropagator(const double mass, const double spacing, int site[4],
			   const int nSmears, const double smearingParameter,
			   const int nSourceSmears,
			   const double sourceSmearingParameter,
			   const int nSinkSmears,
			   const double sinkSmearingParameter,
			   const int solverMethod,
			   const int verbosity)
{
  // Computes the propagator vectors for the 12 spin-colour indices at
  // the given lattice site, using the Dirac operator
  // First off, save the current links and smear all time slices
  GaugeField templinks;
  if (nSmears > 0) {
    templinks = this->links_;
    for (int time = 0; time < this->temporalExtent; time++) {
      this->smearLinks(time, nSmears, smearingParameter);
    }
  }
  // Get the dirac matrix
  if (verbosity > 0)
    cout << "  Generating Dirac matrix..." << flush;
  SparseMatrix<complex<double> > D = this->computeDiracMatrix(mass, spacing);
  if (verbosity > 0)
    cout << " Done!" << endl;
  // Restore the non-smeared gauge field
  if (nSmears > 0)
    this->links_ = templinks;
  // Calculate and return the propagator
  return this->computePropagator(mass, spacing, site, D, nSourceSmears,
				 sourceSmearingParameter, nSinkSmears,
				 sinkSmearingParameter, solverMethod,
				 verbosity);
}
