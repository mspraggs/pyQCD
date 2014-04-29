#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cusp/complex.h>
#include <cusp/linear_operator.h>

typedef cusp::complex<float> Complex;
typedef cusp::linear_operator<cusp::complex<float>, cusp::device_memory> super;
typedef cusp::array1d<Complex, cusp::device_memory> VectorTypeDev;
typedef cusp::array1d<Complex, cusp::host_memory> VectorTypeHost;

void createGammas(Complex* gammas);
void diagonalSpinMatrices(Complex* matrices, Complex factor);
void subtractArray(Complex* y, const Complex* x, const int length);
void addArray(Complex* y, const Complex* x, const int length);
void createLinks(Complex* links, const int L, const int T);
void generateNeighbours(int* indices, const int hopSize, const int L,
			const int T);
void generateBoundaryConditions(Complex* siteBoundaryConditions,
				const int hopSize,
				const Complex* boundaryConditions,
				const int L, const int T);
void setGridAndBlockSize(int& dimBlock, int& dimGrid, int numThreads);
int mod(const int a, const int b);

__device__
int modDev(const int a, const int b);

__global__
void saxpyDev(Complex* y, const Complex* x, const Complex a, const int N);

__global__
void assignDev(Complex* y, const Complex* x, const int N);

__device__
int shiftSiteIndex(const int index, const int direction, const int numHops,
		   const int L, const int T);

template<int numHops>
__device__
Complex computeU(const Complex* links, const int siteIndex, const int mu,
		 const int a, const int b, const int L, const int T);

template<>
__device__
Complex computeU<1>(const Complex* links, const int siteIndex, const int mu,
		    const int a, const int b, const int L, const int T);

#include <utils.tcu>

#endif
