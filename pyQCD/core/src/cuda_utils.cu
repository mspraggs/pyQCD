#include <cuda_utils.h>
#include <cusp/print.h>
#include <stdio.h>

typedef cusp::complex<float> ValueType;

namespace pyQCD
{
  namespace cuda
  {

    __device__
    ValueType gamma(const int mu, const int i, const int j)
    {
      if (mu == 0) {
	if (i == 0 && j == 2)
	  return ValueType(1, 0);
	if (i == 1 && j == 3)
	  return ValueType(1, 0);
	if (i == 2 && j == 0)
	  return ValueType(1, 0);
	if (i == 3 && j == 1)
	  return ValueType(1, 0);
	else
	  return ValueType(0, 0);
      }
      if (mu == 1) {
	if (i == 0 && j == 3)
	  return ValueType(0, -1);
	if (i == 1 && j == 2)
	  return ValueType(0, -1);
	if (i == 2 && j == 1)
	  return ValueType(0, 1);
	if (i == 3 && j == 0)
	  return ValueType(0, 1);
	else
	  return ValueType(0, 0);
      }
      if (mu == 2) {
	if (i == 0 && j == 3)
	  return ValueType(-1, 0);
	if (i == 1 && j == 2)
	  return ValueType(1, 0);
	if (i == 2 && j == 1)
	  return ValueType(1, 0);
	if (i == 3 && j == 0)
	  return ValueType(-1, 0);
	else
	  return ValueType(0, 0);
      }
      if (mu == 3) {
	if (i == 0 && j == 2)
	  return ValueType(0, -1);
	if (i == 1 && j == 3)
	  return ValueType(0, 1);
	if (i == 2 && j == 0)
	  return ValueType(0, 1);
	if (i == 3 && j == 1)
	  return ValueType(0, -1);
	else
	  return ValueType(0, 0);
      }
      return ValueType(0, 0);
    }
 
    __device__
    int mod(int number, const int divisor)
    {
      int ret = number % divisor;
      if (ret < 0)
	ret += divisor;
      return ret;
    }

    __device__
    void addCoords(const int x[4], const int y[4],
		   const int latticeShape[4], int z[4])
    {
      for (int i = 0; i < 4; ++i)
	z[i] = mod(x[i] + y[i], latticeShape[i]);
    }

    __device__
    void getCoords(const int n, int coords[4], const int latticeShape[4])
    {
      int m = n;
      for (int i = 0; i < 4; ++i) {
        coords[3 - i] = mod(m, latticeShape[3 - i]);
	m /= latticeShape[3 - i];
      }
    }

    __device__
    int getIndex(const int coords[4], const int latticeShape[4])
    {
      int ret = 0;

      for (int i = 0; i < 4; ++i) {
	ret *= latticeShape[i];
	ret += mod(coords[i], latticeShape[i]);
      }
      return ret;
    }

    __device__
    void multiplyComplex(const float* x, const float* y, float* z)
    {
      z[0] = x[0] * y[0] - x[1] * y[1];
      z[1] = x[0] * y[1] + x[1] * y[0];
    }

    __global__
    void unprecWilsonKernel(const ValueType* gaugeField, const float mass,
			    const int* latticeShape, const int N,
			    const ValueType* x, ValueType* b)
    {
      int index = blockDim.x * blockIdx.x + threadIdx.x;

      int offsets[8][4] = {{-1,0,0,0},{0,-1,0,0},{0,0,-1,0},{0,0,0,-1},
      	  	           {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};

      if (index < N) {
	int spatialIndex = index / 12;
	int bSpin = (index % 12) / 3;
	int bColour = (index % 12) % 3;
	int coords[4];
	getCoords(spatialIndex, coords, latticeShape);

	// The mass term
	b[index] = 4.0f * x[index];//(4 + mass) * x[index];

	int boundaryCondition = 1;
	// The nearest neighbours
	
	for (int i = 0; i < 8; ++i) {
	  int offsetCoords[4];
	  int dim = i % 4;
	  addCoords(coords, offsets[i], latticeShape, offsetCoords);

	  if (offsetCoords[0] >= latticeShape[0] || offsetCoords[0] < 0)
	    boundaryCondition = -1;

	  int offsetIndex = getIndex(offsetCoords, latticeShape);

	  if (offsetIndex < N)
	    b[index] += x[offsetIndex];

	  if (index == 0) {
	    printf("%d: %d,%d,%d,%d\n",
		   offsetIndex,
		   offsetCoords[0],
		   offsetCoords[1],
		   offsetCoords[2],
		   offsetCoords[3]);
	  }/*
	  for (int j = 0; j < 12; ++j) {
	    int xSpin = j / 3;
	    int xColour = j % 3;

	    int xIndex = (j + 12 * offsetIndex);
	    
	    ValueType spinColourProduct;
	    ValueType fieldElement;
	    ValueType gammaElement;

	    gammaElement = ValueType(1,0);//gamma(dim, bSpin, xSpin);

	    if (i < 4) {
	      int adjointOffset = 1;

	      for (int k = dim + 1; k < 4; ++k)
		adjointOffset *= latticeShape[k];
	      
	      fieldElement = gaugeField[9 * (spatialIndex - adjointOffset)
					+ 3 * xColour + bColour];
	      fieldElement = ValueType(fieldElement.real(),
				       -fieldElement.imag());
	    }
	    else {
	      fieldElement = gaugeField[9 * spatialIndex + 3 * bColour
					+ xColour];
	    }

	    if (bColour == xColour)
	      fieldElement = ValueType(1.0, 0.0);
	    else
	      fieldElement = ValueType(1,0);

	    spinColourProduct = fieldElement * gammaElement;
	    ValueType result = spinColourProduct * x[xIndex];
	    b[index] -= result;// * (float)(0.5 * boundaryCondition);
	    }*/
	}
      }
    }


    class unprecWilsonAction : public cusp::linear_operator<ValueType,cusp::device_memory>
    {
    public:
      typedef cusp::linear_operator<ValueType,cusp::device_memory> super;

      int N;
      ValueType* gaugeField;
      int* latticeShape;
      float mass;

      // constructor
      unprecWilsonAction(int N, const float mass,
			 const cusp::array1d<cusp::complex<float>, hostMem>&
			 cuspGaugeField,
			 const int latticeShape[4]) : super(N,N)
      {
	int fieldSize = 4 * latticeShape[0] * latticeShape[1] * latticeShape[2]
	  * latticeShape[3];
	cudaMalloc((void**) &this->latticeShape, 4 * sizeof(int));
	cudaMemcpy(this->latticeShape, latticeShape, 4 * sizeof(int),
		   cudaMemcpyHostToDevice);

	cudaMalloc((void**) &this->gaugeField, 2 * fieldSize * sizeof(float));
        cudaMemcpy(this->gaugeField,
		   thrust::raw_pointer_cast(&cuspGaugeField[0]),
		   2 * fieldSize * sizeof(float),
		   cudaMemcpyHostToDevice);

        this->mass = mass;
      }

      ~unprecWilsonAction()
      {
	cudaFree(this->gaugeField);
	cudaFree(this->latticeShape);
      }

      // linear operator y = A*x
      template <typename VectorType1,
		typename VectorType2>
      void operator()(const VectorType1& x, VectorType2& y) const
      {
	// obtain a raw pointer to device memory
	const ValueType* x_ptr = thrust::raw_pointer_cast(&x[0]);
	ValueType* y_ptr = thrust::raw_pointer_cast(&y[0]);

	unprecWilsonKernel<<<16,(N + 15) / 16>>>(this->gaugeField,
						 this->mass,
						 this->latticeShape,
						 N, x_ptr, y_ptr);
      }
    };


  
    void createSource(const int spatialIndex, const int spin, const int colour,
		      cusp::array1d<cusp::complex<float>, devMem>& source,
		      cusp::array1d<cusp::complex<float>, devMem>&
		      tempSource)
    {
      int index = colour + 3 * (spin + spatialIndex);
      cusp::blas::fill(tempSource, cusp::complex<float>(0.0, 0.0));
      tempSource[index] = cusp::complex<float>(1.0, 0.0);
    }

  

    void bicgstab(const cusp::array1d<cusp::complex<float>, hostMem>& field,
		  const float mass, const int latticeShape[4],
		  const int spatialIndex,
		  cusp::array2d<cusp::complex<float>, hostMem>& propagator,
		  const int verbosity)
    {
      // Get the size of the Dirac matrix
      int nCols = field.size() / 3;

      unprecWilsonAction devDirac(nCols, mass, field, latticeShape);

      // Create some device arrays to hold the source and solution
      cusp::array1d<cusp::complex<float>, devMem>
	source(nCols, cusp::complex<float>(0, 0));
      cusp::array1d<cusp::complex<float>, devMem>
	solution(nCols, cusp::complex<float>(0, 0));

      // And another load as we'll need duplicates for the
      // multiplication routines
      cusp::array1d<cusp::complex<float>, devMem>
	tempSource(nCols, cusp::complex<float>(0, 0));
      cusp::array1d<cusp::complex<float>, devMem>
	tempSolution(nCols, cusp::complex<float>(0, 0));

      // Create a temporary propagator variable for fast access
      cusp::array2d<cusp::complex<float>, devMem>
	tempPropagator(nCols, 12, cusp::complex<float>(0, 0));

      unprecWilsonAction devDirac2(100, mass, field, latticeShape);

      cusp::array1d<cusp::complex<float>, devMem> temp(100, cusp::complex<float>(0, 0));
      cusp::array1d<cusp::complex<float>, devMem> temp2(100, cusp::complex<float>(0, 0));
      createSource(0, 0, 0, temp, temp);
      cusp::multiply(devDirac2, temp, temp2);
      cusp::array1d<cusp::complex<float>, hostMem> temp3
	= temp2;
      cusp::print(temp3);
      
      // Loop through all spins and colours and do the inversions
      for (int i = 0; i < 4; ++i) {
	for (int j = 0; j < 3; ++j) {
	  if (verbosity > 0)
	    std::cout << "  Inverting for spin " << i
		      << " and colour " << j << "..." << std::endl;
	  // Create the source using the smearing operator
	  createSource(spatialIndex, i, j, source,
		       source);
	  // Set up the monitor for use in the solver
	  cusp::default_monitor<cusp::complex<float> >
	    monitor(source, 1000, 0, 1e-8);
	  
	  // Do the inversion
	  cusp::krylov::bicgstab(devDirac, solution, source, monitor);
	  // Create a view to the relevant column of the propagator
	  cusp::array2d<cusp::complex<float>, devMem>::column_view
	    propagatorView = tempPropagator.column(j + 3 * i);
	  // Copy the solution to the propagator output
	  cusp::copy(solution, propagatorView);
	  if (verbosity > 0)
	    std::cout << "  -> Inversion reached tolerance of " 
		      << monitor.residual_norm() << " in "
		      << monitor.iteration_count() << " iterations."
		      << std::endl;
	}
      }
      // Move the propagator back into main memory
      if (verbosity > 0)
	std::cout << "  Transferring propagators to main memory..."
		  << std::flush;
      propagator = tempPropagator;
      if (verbosity > 0)
	std::cout << " Done!" << std::endl;
    }

  
  
    void cg(const complexHybridHost& hostDiracDiracAdjoint,
	    const complexHybridHost& hostDiracAdjoint,
	    const complexHybridHost& hostSourceSmear,
	    const complexHybridHost& hostSinkSmear,
	    const int spatialIndex,
	    cusp::array2d<cusp::complex<float>, hostMem>& propagator,
	    const int verbosity)
    {
      // Get the size of the Dirac matrix
      int nCols = hostDiracDiracAdjoint.num_cols;
      // Transfer the Dirac and smearing matrices to the device.
      if (verbosity > 0)
	std::cout << "  Transferring matrices to device..." << std::flush;
      complexHybridDev devM = hostDiracDiracAdjoint;
      complexHybridDev devDadj = hostDiracAdjoint;
      complexHybridDev devSourceSmear = hostSourceSmear;
      complexHybridDev devSinkSmear = hostSinkSmear;
      if (verbosity > 0)
	std::cout << " Done!" << std::endl;

      // Create some device arrays to hold the source and solution
      cusp::array1d<cusp::complex<float>, devMem>
	source(nCols, cusp::complex<float>(0, 0));
      cusp::array1d<cusp::complex<float>, devMem>
	solution(nCols, cusp::complex<float>(0, 0));

      // And another load as we'll need duplicates for the
      // multiplication routines
      cusp::array1d<cusp::complex<float>, devMem>
	tempSource(nCols, cusp::complex<float>(0, 0));
      cusp::array1d<cusp::complex<float>, devMem>
	tempSolution(nCols, cusp::complex<float>(0, 0));

      // Create a temporary propagator variable for fast access
      cusp::array2d<cusp::complex<float>, devMem>
	tempPropagator(nCols, 12, cusp::complex<float>(0, 0));

      // Create the preconditioner
      cusp::precond::diagonal<cusp::complex<float>, devMem>
	preconditioner(devM);

      // Loop through all spins and colours and do the inversions
      for (int i = 0; i < 4; ++i) {
	for (int j = 0; j < 3; ++j) {
	  if (verbosity > 0)
	    std::cout << "  Inverting for spin " << i
		      << " and colour " << j << "..." << std::endl;
	  // Create the source using the smearing operator
	  createSource(spatialIndex, i, j, source,
		       tempSource);
	  // Set up the monitor for use in the solver
	  cusp::default_monitor<cusp::complex<float> >
	    monitor(source, 1000, 0, 1e-8);
	  
	  // Do the inversion
	  cusp::krylov::cg(devM, solution, source, monitor,
			   preconditioner);
	  // Smear at the sink
	  cusp::multiply(devDadj, solution, tempSolution);
	  
	  cusp::multiply(devSinkSmear, tempSolution, solution);
	  // Create a view to the relevant column of the propagator
	  cusp::array2d<cusp::complex<float>, devMem>::column_view
	    propagatorView = tempPropagator.column(j + 3 * i);
	  // Copy the solution to the propagator output
	  cusp::copy(solution, propagatorView);
	  if (verbosity > 0)
	    std::cout << "  -> Inversion reached tolerance of " 
		      << monitor.residual_norm() << " in "
		      << monitor.iteration_count() << " iterations."
		      << std::endl;
	}
      }
      // Move the propagator back into main memory
      if (verbosity > 0)
	std::cout << "  Transferring propagators to main memory..."
		  << std::flush;
      propagator = tempPropagator;
      if (verbosity > 0)
	std::cout << " Done!" << std::endl;
    }
  }
}
