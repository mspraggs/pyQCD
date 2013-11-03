#include <cuda_utils.h>

namespace pyQCD
{
  namespace cuda
  {
    __constant__
    float gammas[128] = {0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 1, 0,
			 1, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0,
			   
			 0, 0, 0, 0, 0, 0, 0, -1,
			 0, 0, 0, 0, 0, -1, 0, 0,
			 0, 0, 0, 1, 0, 0, 0, 0,
			 0, 1, 0, 0, 0, 0, 0, 0,

			 0, 0, 0, 0, 0, 0, -1, 0,
			 0, 0, 0, 0, 1, 0, 0, 0,
			 0, 0, 1, 0, 0, 0, 0, 0,
			 -1, 0, 0, 0, 0, 0, 0, 0,

			 0, 0, 0, 0, 0, -1, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 1,
			 0, 1, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, -1, 0, 0, 0, 0};
    
    __device__
    int mod(int number, const int divisor)
    {
      int ret = number % divisor;
      if (ret < 0)
	ret += divisor;
      return ret;
    }

    __device__
    void addCoords(const int x[4], const int y[4], int z[4])
    {
      for (int i = 0; i < 4; ++i)
	z[i] = x[i] + y[i];
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
    void unprecWilsonKernel(const float* gaugeField, const float mass,
			    const int* latticeShape, const int N, const float* x,
			    float* b)
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
	b[2 * index] = (4 + mass) * x[2 * index];
	b[2 * index + 1] = (4 + mass) * x[2 * index + 1];

	int boundaryCondition = 1;
	// The nearest neighbours
	for (int i = 0; i < 8; ++i) {
	  int offsetCoords[4];
	  int dim = i % 4;
	  addCoords(coords, offsets[i], offsetCoords);

	  if (offsetCoords[0] >= latticeShape[0] || offsetCoords[0] < 0)
	    boundaryCondition = -1;

	  int offsetIndex = getIndex(offsetCoords, latticeShape);

	  for (int j = 0; j < 12; ++j) {
	    int xSpin = j / 3;
	    int xColour = j % 3;

	    int xIndex = 2 * (j + 12 * offsetIndex);
	    
	    float spinColourProduct[2];
	    float fieldElement[2];
	    float gammaElement[2];

	    gammaElement[0] = gammas[32 * dim + 8 * bSpin + 2 * xSpin];
	    gammaElement[1] = gammas[32 * dim + 8 * bSpin + 2 * xSpin + 1];

	    if (i < 4) {
	      int adjointOffset = 1;

	      for (int k = dim + 1; k < 4; ++k)
		adjointOffset *= latticeShape[k];
	      
	      fieldElement[0] = gaugeField[18 * (spatialIndex - adjointOffset)
					   + 6 * xColour + 2 * bColour];
	      fieldElement[1] = -gaugeField[18 * (spatialIndex - adjointOffset)
					    + 6 * xColour + 2 * bColour + 1];
	    }
	    else {
	      fieldElement[0] = gaugeField[18 * spatialIndex + 6 * bColour
					   + 2 * xColour];
	      fieldElement[1] = gaugeField[18 * spatialIndex + 6 * bColour
					   + 2 * xColour + 1];
	    }

	    multiplyComplex(fieldElement, gammaElement, spinColourProduct);
	    float result[2];
	    multiplyComplex(spinColourProduct, &x[xIndex], result);
	    b[2 * index] -= 0.5 * boundaryCondition * result[0];
	    b[2 * index + 1] -= 0.5 * boundaryCondition * result[1];
	  }
	}
      }
    }


    class unprecWilsonAction : public cusp::linear_operator<float,cusp::device_memory>
    {
    public:
      typedef cusp::linear_operator<float,cusp::device_memory> super;

      int N;
      float* gaugeField;
      int* latticeShape;
      float mass;

      // constructor
      unprecWilsonAction(int N, float mass,
			 cusp::array1d<cusp::complex<float>, hostMem>&
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
	const float* x_ptr = thrust::raw_pointer_cast((float*)&x[0]);
	float* y_ptr = thrust::raw_pointer_cast((float*)&y[0]);

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


    int getMem()
    {
      size_t free;
      size_t total;
      cuMemGetInfo(&free, &total);

      return free;
    }

  

    void bicgstab(const cusp::array1d<cusp::complex<float>, hostMem>& field,
		  const float mass, const int latticeShape[4],
		  const int spatialIndex,
		  cusp::array2d<cusp::complex<float>, hostMem>& propagator,
		  const int verbosity)
    {
      // Get the size of the Dirac matrix
      int nCols = field.size() / 6;

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
	  createSource(spatialIndex, i, j, devSourceSmear, source,
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
