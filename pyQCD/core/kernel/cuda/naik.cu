#include <naik.cu>

Naik::Naik(const float mass, const int L, const int T,
	   const bool precondition, const bool hermitian,
	   const Complex* boundaryConditions, Complex* links,
	   const bool copyLinks = true)
  : LinearOperator(L, T, precondition, hermitian, links, copyLinks)
{
  this->mass_ = mass;

  // First generate the neighbours
  int numSites = this->N / 12;
  int size = 8 * numSites * sizeof(int);
  int* hostIndices = (int*) malloc(size);
  cudaMalloc((void**) &this->neighbourIndices_, size);
  generateNeighbours(hostIndices, 1, L, T);
  cudaMemcpy(this->neighbourIndices_, hostIndices, size, cudaMemcpyHostToDevice);

  // And generate the next to nearest neighbours
  cudaMalloc((void**) &this->nextNextNeighbourIndices_, size);
  generateNeighbours(hostIndices, 3, L, T);
  cudaMemcpy(this->nextNextNeighbourIndices_, hostIndices, size,
	     cudaMemcpyHostToDevice);
  free(hostIndices);

  // Now generate the spin structures needed. There are eight, one for each
  // of the hops that have to be performed. The array has to be flattened, as
  // CUDA doesn't like 2D arrays
  size = 8 * 16 * sizeof(Complex); // Eight matrices, 16 complex numbers each
  Complex hostGammas[64]; // We'll store temporary gamma matrices here
  Complex hostSpinStructures[128]; // We'll set up the spin structures here
  // before sending them to the device
  createGammas(hostGammas); // Initialise the gamma matrices

  // Now set the spin structes up with the identity, which we'll then subtract
  // and add the gamma matrices from/to.
  diagonalSpinMatrices(hostSpinStructures, Complex(1.0, 0.0));
  diagonalSpinMatrices(hostSpinStructures + 64, Complex(1.0, 0.0));
  subtractArray(hostSpinStructures, hostGammas, 64); // 1 - gamma_mu
  addArray(hostSpinStructures + 64, hostGammas, 64); // 1 + gamma_mu
  // Now send everything to the device
  cudaMalloc((void**) &this->spinStructures_, size);
  cudaMemcpy(this->spinStructures_, hostSpinStructures, size,
	     cudaMemcpyHostToDevice);

  // Now set the spin structes up with the identity, which we'll then subtract
  // and add the gamma matrices from/to.
  diagonalSpinMatrices(hostSpinStructures, Complex(3.0, 0.0));
  diagonalSpinMatrices(hostSpinStructures + 64, Complex(3.0, 0.0));
  subtractArray(hostSpinStructures, hostGammas, 64); // 3 - gamma_mu
  addArray(hostSpinStructures + 64, hostGammas, 64); // 3 + gamma_mu
  // Now send everything to the device
  cudaMalloc((void**) &this->naikSpinStructures_, size);
  cudaMemcpy(this->naikSpinStructures_, hostSpinStructures, size,
	     cudaMemcpyHostToDevice);

  // Set up the boundary conditions for one hop
  size = 8 * numSites * sizeof(Complex);
  Complex* hostBoundaryConditions = (Complex*) malloc(size);
  generateBoundaryConditions(hostBoundaryConditions, 1,
			     boundaryConditions, L, T);
  cudaMalloc((void**) &this->boundaryConditions_, size);
  cudaMemcpy(this->boundaryConditions_, hostBoundaryConditions, size,
	     cudaMemcpyHostToDevice);
  // Set up the boundary conditions for two hops
  generateBoundaryConditions(hostBoundaryConditions, 3,
			     boundaryConditions, L, T);
  cudaMalloc((void**) &this->naikBoundaryConditions_, size);
  cudaMemcpy(this->naikBoundaryConditions_, hostBoundaryConditions, size,
	     cudaMemcpyHostToDevice);
  free(hostBoundaryConditions);
}



Naik::~Naik()
{
  cudaFree(this->neighbourIndices_);
  cudaFree(this->spinStructures_);
  cudaFree(this->nextNextNeighbourIndices_);
  cudaFree(this->naikSpinStructures_);
  cudaFree(this->boundaryConditions_);
  cudaFree(this->naikBoundaryConditions_);
}



void Naik::apply(Complex* y, const Complex* x) const
{  
  int dimBlock;
  int dimGrid;

  setGridAndBlockSize(dimBlock, dimGrid, this->N);

  diagonalKernel<<<dimGrid,dimBlock>>>(y, x, 4 + this->mass_,
				       this->L, this->T);
  
  hoppingKernel<1><<<dimGrid,dimBlock>>>(y, x, this->links_, 
					 this->spinStructures_,
					 this->neighbourIndices_,
					 this->boundaryConditions_,
					 -9.0 / 16.0, this->L, this->T);
  
  hoppingKernel<3><<<dimGrid,dimBlock>>>(y, x, this->links_, 
					 this->naikSpinStructures_,
					 this->nextNextNeighbourIndices_,
					 this->naikBoundaryConditions_,
					 1.0 / 16.0, this->L, this->T);
}



void Naik::applyHermitian(Complex* y, const Complex* x) const
{
  this->apply(y, x);
  int dimBlock;
  int dimGrid;
  setGridAndBlockSize(dimBlock, dimGrid, this->N);
  applyGamma5<<<dimGrid,dimBlock>>>(y, y, this->L, this->T);
}



void Naik::makeHermitian(Complex* y, const Complex* x) const
{
  int dimBlock;
  int dimGrid;
  setGridAndBlockSize(dimBlock, dimGrid, this->N);
  applyGamma5<<<dimGrid,dimBlock>>>(y, x, this->L, this->T);
}
