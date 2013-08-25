#include <random.hpp>
#include <iostream>

Random::Random()
{
  // Default constructor
  // Find out how many generators we need, putting this to 1 if we're not
  // using OpenMP
#if defined(_OPENMP)
  this->nThreads = omp_get_max_threads();
#else
  this->nThreads = 1;
#endif
  
  // Create a series of generators, one for each thread
  for (int i = 0; i < this->nThreads; ++i) {
    this->generators.push_back(mt19937(time(0) + i));
  }
}



Random::~Random()
{
  // Destructor
}



void Random::setSeed(const int seed)
{
  // Specify a custom random number seed  
  // Reset the generators, one for each thread
  for (int i = 0; i < this->nThreads; ++i) {
    this->generators[i] = mt19937(seed + i);
  }
}



double Random::generateReal()
{
  // Generates a random real number from the corresponding generator

  // Determine which generator to use
#if defined(_OPENMP)
  int currentThread = omp_get_thread_num();
#else
  int currentThread = 0;
#endif

  // Declare a uniform distribution
  uniform_real<> tempRealDistribution(0, 1);
  // Now declare the variate generator
  variate_generator<mt19937&, uniform_real<> >
    tempRealGenerator(this->generators[currentThread], tempRealDistribution);
  // Generate and return the real number
  return tempRealGenerator();
}



int Random::generateInt()
{
  // Generates an random integer number from the corresponding generator

  // Determine which generator to use
#if defined(_OPENMP)
  int currentThread = omp_get_thread_num();
#else
  int currentThread = 0;
#endif
  // Declare a uniform distribution
  uniform_int<> tempIntDistribution(0, 199);
  // Declare the specific variate generator
  variate_generator<mt19937&, uniform_int<> >
    tempIntGenerator(this->generators[currentThread], tempIntDistribution);
  // Generate and return the integer
  return tempIntGenerator();
}
