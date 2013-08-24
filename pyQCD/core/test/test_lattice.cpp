#define BOOST_TEST_MODULE Lattice test
#include <lattice.hpp>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <complex>

using namespace Eigen;

bool areEqual(const double x, const double y, const double precision)
{
  if (fabs(x - y) < precision)
    return true;
  else
    return false;
}

bool areEqual(const complex<double> x, const complex<double> y,
	      const double precision)
{
  if (!areEqual(x.real(), y.real(), precision))
    return false;
  else if (!areEqual(x.imag(), y.imag(), precision))
    return false;
  else
    return true;
}

bool areEqual(const Matrix3cd& A, const Matrix3cd& B, const double precision)
{
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (!areEqual(A(i, j), B(i, j), precision))
	return false;

  return true;
}

BOOST_AUTO_TEST_CASE( gluonic_measurements_test )
{
  Lattice lattice;

  int site[4] = {0, 0, 0, 0};

  for (int i = 1; i < 4; ++i) {
    for (int j = 0; j < i; ++j) {
      BOOST_CHECK_EQUAL(lattice.computePlaquette(site, i, j), 1.0);
      BOOST_CHECK_EQUAL(lattice.computeRectangle(site, i, j), 1.0);
      BOOST_CHECK_EQUAL(lattice.computeTwistedRectangle(site, i, j), 1.0);
    }
  }

  BOOST_CHECK_EQUAL(lattice.computeAveragePlaquette(), 1.0);
  BOOST_CHECK_EQUAL(lattice.computeAverageRectangle(), 1.0);

  BOOST_CHECK_EQUAL(lattice.computeAverageWilsonLoop(1, 1), 1.0);
  BOOST_CHECK_EQUAL(lattice.computeAverageWilsonLoop(1, 1, 1, 0.5), 1.0);
}

BOOST_AUTO_TEST_CASE( update_test )
{
  Lattice lattice(8, 8, 5.5, 1.0, 0, 10, 0, 1, 4);
  lattice.thermalize();
      
  BOOST_CHECK(lattice.computeAveragePlaquette() < 0.51 &&
	      lattice.computeAveragePlaquette() > 0.49);
      
  BOOST_CHECK(lattice.computeAverageRectangle() < 0.27 &&
	      lattice.computeAverageRectangle() > 0.25);

  lattice = Lattice(8, 8, 5.5, 1.0, 0, 60, 1, 1, 4);
  lattice.thermalize();
      
  BOOST_CHECK(lattice.computeAveragePlaquette() < 0.51 &&
	      lattice.computeAveragePlaquette() > 0.49);
      
  BOOST_CHECK(lattice.computeAverageRectangle() < 0.27 &&
	      lattice.computeAverageRectangle() > 0.25);

  lattice = Lattice(8, 8, 5.5, 1.0, 0, 10, 0, 0, 4);
  lattice.thermalize();
      
  BOOST_CHECK(lattice.computeAveragePlaquette() < 0.51 &&
	      lattice.computeAveragePlaquette() > 0.49);
      
  BOOST_CHECK(lattice.computeAverageRectangle() < 0.27 &&
	      lattice.computeAverageRectangle() > 0.25);

  lattice = Lattice(8, 8, 5.5, 1.0, 0, 60, 1, 0, 4);
  lattice.thermalize();
      
  BOOST_CHECK(lattice.computeAveragePlaquette() < 0.51 &&
	      lattice.computeAveragePlaquette() > 0.49);
      
  BOOST_CHECK(lattice.computeAverageRectangle() < 0.27 &&
	      lattice.computeAverageRectangle() > 0.25);
}
