

class Lattice
{
public:
  Lattice(const int n = 8,
	  const double beta = 5.5,
	  const int Ncor = 50,
	  const int Ncf = 1000,
	  const double eps = 0.24);

  ~Lattice();
  P(const int site[4], const in mu, const int nu);
  Pav();
  Si(const int link[5]);
  randomSU3();

private:
  int n, Ncor, Ncf;
  double beta, eps;
}
