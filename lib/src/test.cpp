#include "lattice.cpp"
#include <execinfo.h>
#include <signal.h>
#include <ctime>

void handler(int sig) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  cout << "Error: signal" << sig << endl;
  backtrace_symbols_fd(array, size, 2);
  exit(1);
}

int main()
{
  //struct timeval start, end;
  //signal(SIGSEGV, handler);
  Lattice L = Lattice();
  cout << "Thermalizing lattice..." << endl;
  //L.thermalize();
  cout << "Done!" << endl;
  
  clock_t start;
  start = clock();
  
  //int site[4] = {0,0,0,0};
  //SparseMatrix<complex<double> > D = L.DiracMatrix(1);
  //VectorXcd prop = L.Propagator(1,site,0,0);
  for (int i = 0; i < 1e9; i++)
    int j = lattice::mod(-400, 8);
  
  double t = double(clock() - start) / double(CLOCKS_PER_SEC);
  cout << "Execution time: " << t << endl;

  //cout << prop << endl;
  
  return 0;
}
