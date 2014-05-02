#ifndef BASE_H
#define BASE_H

#include <cusp/complex.h>
#include <cusp/array2d.h>
#include <cusp/array1d.h>

typedef cusp::complex<float> Complex;
typedef cusp::array2d<Complex, cusp::host_memory> PropagatorTypeHost;
typedef cusp::array1d<Complex, cusp::host_memory> VectorTypeHost;

#endif
