#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <utils.h>

__global__
void diagonalKernel(Complex* y, const Complex* x, const Complex scaling,
		    const int L, const int T);

__global__
void applyGamma5(Complex* y, const Complex* x, const int L, const int T);

__global__
void applyPplus(Complex* y, const Complex* x, const int L, const int T);

__global__
void applyPminus(Complex* y, const Complex* x, const int L, const int T);

template<int numHops>
__global__ 
void hoppingKernel(Complex* y, const Complex* x, const Complex* links,
		   const Complex* gammas, const int* neighbourIndices,
		   const Complex* boundaryConditions, const Complex scaling,
		   const int L, const int T);

template<int numHops>
__global__ 
void precHoppingKernel(Complex* y, const Complex* x, const Complex* links,
		       const Complex* gammas, const int* neighbourIndices,
		       const int* siteIndices,
		       const Complex* boundaryConditions,
		       const Complex scaling, const int L, const int T);

template<int numHops>
__global__ 
void hoppingKernel3d(Complex* y, const Complex* x, const Complex* links,
		     const Complex* gammas, const int* neighbourIndices,
		     const Complex* boundaryConditions, const Complex scaling,
		     const int L, const int T);

#include <kernels.tcu>

#endif
