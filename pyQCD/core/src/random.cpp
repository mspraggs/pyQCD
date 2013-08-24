#include <random.hpp>
#include <iostream>

Random::Random()
{
  // Default constructor
  
#if defined(_OPENMP)
  this->nThreads = omp_get_max_threads();
#else
  this->nThreads = 1;
#endif
  
  //this->generators.resize(this->nThreads);
  //this->uniformReals.resize(this->nThreads);
  //this->uniformInts.resize(this->nThreads);
  //this->realGenerators.resize(this->nThreads);
  //this->intGenerators.resize(this->nThreads);
  
  for (int i = 0; i < this->nThreads; ++i) {
    this->generators.push_back(mt19937(time(0) + i));
  }
}



Random::~Random()
{
  // Destructor
}



double Random::generateReal()
{
  // Generates a random real number from the corresponding generator

#if defined(_OPENMP)
  int currentThread = omp_get_thread_num();
#else
  int currentThread = 0;
#endif

  uniform_real<> tempRealDist(0, 1);

  variate_generator<mt19937&, uniform_real<> >
    tempRealGen(this->generators[currentThread], tempRealDist);

  return tempRealGen();
}



int Random::generateInt()
{
  // Generates a random real number from the corresponding generator

#if defined(_OPENMP)
  int currentThread = omp_get_thread_num();
#else
  int currentThread = 0;
#endif

  uniform_int<> tempIntDist(0, 199);

  variate_generator<mt19937&, uniform_int<> >
    tempIntGen(this->generators[currentThread], tempIntDist);

  return tempIntGen();
}
