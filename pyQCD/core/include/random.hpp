#include <boost/random.hpp>
#include <vector>
#include <omp.h>

using namespace boost;
using namespace std;

class Random
{
  // Provides thread safety for the boost random number generator by creating a
  // generator for each thread
public:
  Random();
  ~Random();

  double generateReal();
  int generateInt();
  void setSeed(const int seed);

private:
  int nThreads;
  vector<mt19937> generators;

  Random(const Random& rng);
  Random& operator=(const Random& rng);
};
