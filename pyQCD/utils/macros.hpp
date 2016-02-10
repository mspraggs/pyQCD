#ifndef PYQCD_MACROS_HPP
#define PYQCD_MACROS_HPP

/* This file provides macros to help neaten up code and encapsulate various
 * preprocessor bits and pieces, particularly those relating to the number of
 * dimensions and colours in the simulation.
 */

#include <iostream>

// Custom assert command - cython can process this.
#ifndef NDEBUG
#define pyQCDassert(expr, exception)                            \
if (not (expr)) {                                               \
  std::cout << "Assertion " << #expr << " failed" << std::endl; \
  throw exception;                                              \
}
#else
#define pyQCDassert(expr, exception)
#endif

#endif