#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <complex>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <unsupported/Eigen/MatrixFunctions>
#include "gil.cpp"
#include <omp.h>

typedef vector<Matrix3cd, aligned_allocator<Matrix3cd> > Sub4Field;
typedef vector<Sub4Field> Sub3Field;
typedef vector<Sub3Field> Sub2Field;
typedef vector<Sub2Field> SubField;
typedef vector<SubField> GaugeField;
typedef Triplet<complex<double> > Tlet;

class Lattice
{
public:
  Lattice(const int n = 8,
	  const double beta = 5.5,
	  const int Ncor = 50,
	  const int Ncf = 1000,
	  const double eps = 0.24,
	  const double a = 0.25,
	  const double smear_eps = 0.3,
	  const double u0 = 1,
	  const int action = 0);
  Lattice(const Lattice& L);
  double init_u0();

  ~Lattice();
  Matrix3cd calcPath(const vector<vector<int> > path);
  Matrix3cd calcLine(const int start[4], const int finish[4]);
  double W(const int c1[4], const int c2[4], const int n_smears = 0);
  double W(const int c[4], const int r, const int t, const int dim, const int n_smears = 0);
  double Wav(const int r, const int t, const int n_smears = 0);
  double P(const int site[4], const int mu, const int nu);
  double R(const int site[4], const int mu, const int nu);
  double T(const int site[4],const int mu, const int nu);
  double Pav();
  double Rav();
  double (Lattice::*Si)(const int link[5]);
  Matrix3cd randomSU3();
  void thermalize();
  void nextConfig();
  void runThreads(const int size, const int n_updates, const int remainder);
  void updateSchwarz(const int size, const int n_times);
  void update();
  void updateSegment(const int i, const int j, const int k, const int l, const int size, const int n_updates);
  void printL();
  Matrix3cd link(const int link[5]);
  void smear(const int time, const int n_smears);
  Matrix3cd Q(const int link[5]);
  py::list getLink(const int i, const int j, const int k, const int l, const int m) const;
  py::list getRandSU3(const int i) const;

  SparseMatrix<complex<double> > DiracMatrix(const double mass);
  VectorXcd Propagator(const double mass, int site[4], const int alpha, const int a);

  int Ncor, Ncf, n;

protected:
  double beta, eps, a, u0, smear_eps;
  int nupdates, action;
  double SiW(const int link[5]);
  double SiImpR(const int link[5]);
  double SiImpT(const int link[5]);
  GaugeField links;
  Sub4Field randSU3s;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
};
