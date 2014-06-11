
__global__
void diagonalKernel(Complex* y, const Complex* x, const Complex scaling,
		    const int L, const int T)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  int N = 12 * L * L * L * T;

  if (index < N)
    y[index] = scaling * x[index];
}

__global__
void applyGamma5(Complex* y, const Complex* x, const int L, const int T)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  int N = 12 * L * L * L * T;

  if (index < N) {

    int alpha = (index % 12) / 3;

    y[index] = (alpha < 2) ? x[index] : -x[index];
  }
}

__global__
void applyPminus(Complex* y, const Complex* x, const int L, const int T)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  int N = 12 * L * L * L * T;

  if (index < N) {

    int alpha = (index % 12) / 3;

    y[index] = (alpha < 2) ? 0.0 : x[index];
  }
}

__global__
void applyPplus(Complex* y, const Complex* x, const int L, const int T)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  int N = 12 * L * L * L * T;

  if (index < N) {

    int alpha = (index % 12) / 3;

    y[index] = (alpha < 2) ? x[index] : 0.0;
  }
}

template<int numHops>
__global__ 
void hoppingKernel(Complex* y, const Complex* x, const Complex* links,
		   const Complex* gammas, const int* neighbourIndices,
		   const Complex* boundaryConditions, const Complex scaling,
		   const int L, const int T)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  int N = 12 * L * L * L * T;

  if (index < N) {
    int siteIndex = index / 12;
    int alpha = (index % 12) / 3;
    int a = index % 3;

    for (int mu = 0; mu < 4; ++mu) {
      int siteBehindIndex = neighbourIndices[8 * siteIndex + mu];
      int siteAheadIndex = neighbourIndices[8 * siteIndex + mu + 4];

      for (int j = 0; j < 12; ++j) {
	int beta = j / 3;
	int b = j % 3;
      
	Complex Up = computeU<numHops>(links, siteIndex, mu, a, b, L, T);
	Complex Um
	  = cusp::conj(computeU<numHops>(links, siteBehindIndex, mu, b, a,
					 L, T));

	Complex temp
	  = Um * gammas[16 * mu + 4 * alpha + beta]
	  * boundaryConditions[8 * siteIndex + mu]
	  * x[12 * siteBehindIndex + j];
	temp
	  += Up * gammas[64 + 16 * mu + 4 * alpha + beta]
	  * boundaryConditions[8 * siteIndex + mu + 4]
	  * x[12 * siteAheadIndex + j];

	y[index] += scaling * temp;
      }
    }
  }
}

template<int numHops>
__global__ 
void precHoppingKernel(Complex* y, const Complex* x, const Complex* links,
		       const Complex* gammas, const int* neighbourIndices,
		       const int* siteIndices,
		       const Complex* boundaryConditions,
		       const Complex scaling, const int L, const int T)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  int N = 6 * L * L * L * T;

  if (index < N) {
    int siteIndex = index / 12;
    int alpha = (index % 12) / 3;
    int a = index % 3;

    for (int mu = 0; mu < 4; ++mu) {
      int siteBehindIndex = neighbourIndices[8 * siteIndex + mu];
      int siteAheadIndex = neighbourIndices[8 * siteIndex + mu + 4];

      for (int j = 0; j < 12; ++j) {
	int beta = j / 3;
	int b = j % 3;
      
	Complex Up = computeU<numHops>(links, siteIndices[siteIndex], mu, a, b,
				       L, T);
	Complex Um
	  = cusp::conj(computeU<numHops>(links, siteBehindIndex, mu, b, a,
					 L, T));
	
	y[index]
	  += scaling * Um * gammas[16 * mu + 4 * alpha + beta]
	  * boundaryConditions[8 * siteIndices[siteIndex] + mu]
	  * x[12 * (siteBehindIndex / 2) + j];
	y[index]
	  += scaling * Up * gammas[64 + 16 * mu + 4 * alpha + beta]
	  * boundaryConditions[8 * siteIndices[siteIndex] + mu + 4]
	  * x[12 * (siteAheadIndex / 2) + j];
      }
    }
  }
}

template<int numHops>
__global__ 
void hoppingKernel3d(Complex* y, const Complex* x, const Complex* links,
		     const Complex* gammas, const int* neighbourIndices,
		     const Complex* boundaryConditions, const Complex scaling,
		     const int L, const int T)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  int N = 12 * L * L * L * T;

  if (index < N) {
    int siteIndex = index / 12;
    int alpha = (index % 12) / 3;
    int a = index % 3;

    for (int mu = 1; mu < 4; ++mu) {
      int siteBehindIndex = neighbourIndices[8 * siteIndex + mu];
      int siteAheadIndex = neighbourIndices[8 * siteIndex + mu + 4];

      for (int j = 0; j < 12; ++j) {
	int beta = j / 3;
	int b = j % 3;
      
	Complex Up = computeU<numHops>(links, siteIndex, mu, a, b, L, T);
	Complex Um
	  = cusp::conj(computeU<numHops>(links, siteBehindIndex, mu, b, a,
					 L, T));
	
	y[index]
	  += scaling * Um * gammas[16 * mu + 4 * alpha + beta]
	  * boundaryConditions[8 * siteIndex + mu]
	  * x[12 * siteBehindIndex + j];
	y[index]
	  += scaling * Up * gammas[64 + 16 * mu + 4 * alpha + beta]
	  * boundaryConditions[8 * siteIndex + mu + 4]
	  * x[12 * siteAheadIndex + j];
      }
    }
  }
}
