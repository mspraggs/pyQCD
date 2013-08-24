#include <boost/random.hpp>
#include <vector>
#include <omp.h>

using namespace boost;
using namespace std;

class Random
{
public:
  Random();
  ~Random();

  double generateReal();
  int generateInt();
  
private:
  int nThreads;
  vector<mt19937> generators;
};
