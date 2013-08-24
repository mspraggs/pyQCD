#define BOOST_TEST_MODULE Lattice test
#include <lattice.hpp>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <complex>
#include <cstdlib>

using namespace Eigen;
using namespace std;

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

complex<double> randomComplexNumber()
{
  srand(time(0));

  double x = double(rand()) / double(RAND_MAX);
  double y = double(rand()) / double(RAND_MAX);

  return complex<double>(x, y);
}

BOOST_AUTO_TEST_CASE( gluonic_measurements_test )
{
  Lattice lattice;
  complex<double> randC = randomComplexNumber();
  Matrix3cd randomDiagonalMatrix = randC * Matrix3cd::Identity();
  
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
	for (int l = 0; l < 4; ++l) {
	  for (int m = 0; m < 4; ++m) {
	    int tempLink[5] = {i, j, k, l, m};
	    lattice.setLink(tempLink, randomDiagonalMatrix);
	  }
	}
      }
    }
  }

  double plaquetteVal = pow(abs(randC), 4);
  double rectangleVal = pow(abs(randC), 6);
  double twistedRectangleVal = pow(abs(randC), 8);

  int site[4] = {0, 0, 0, 0};

  // Checking all plaquettes, rectangles and twisted rectangles
  for (int i = 1; i < 4; ++i) {
    for (int j = 0; j < i; ++j) {
      BOOST_CHECK(areEqual(lattice.computePlaquette(site, i, j),
			   plaquetteVal, 100 * DBL_EPSILON));
      BOOST_CHECK(areEqual(lattice.computeRectangle(site, i, j),
			   rectangleVal, 100 * DBL_EPSILON));
      BOOST_CHECK(areEqual(lattice.computeTwistedRectangle(site, i, j),
			   twistedRectangleVal, 100 * DBL_EPSILON));
    }
  }
  // Checking average plaquette and rectangle
  BOOST_CHECK(areEqual(lattice.computeAveragePlaquette(),
		       plaquetteVal, 100 * DBL_EPSILON));
  BOOST_CHECK(areEqual(lattice.computeAverageRectangle(),
		       rectangleVal, 100 * DBL_EPSILON));
  // Checking average Wilson loops
  BOOST_CHECK(areEqual(lattice.computeAverageWilsonLoop(1, 1),
		       plaquetteVal, 100 * DBL_EPSILON));
  BOOST_CHECK(areEqual(lattice.computeAverageWilsonLoop(1, 1, 1, 0.5),
		       plaquetteVal, 100 * DBL_EPSILON));
}

BOOST_AUTO_TEST_CASE( update_test )
{
  int linkCoords[5] = {0, 0, 0, 0, 0};

  // Checking parallel heatbath updates
  Lattice lattice(8, 8, 5.5, 1.0, 0, 10, 0, 1, 4);
  lattice.thermalize();
  // Check basic observables
  BOOST_CHECK(lattice.computeAveragePlaquette() < 0.51 &&
	      lattice.computeAveragePlaquette() > 0.49);   
  BOOST_CHECK(lattice.computeAverageRectangle() < 0.27 &&
	      lattice.computeAverageRectangle() > 0.25);
  // Check unitarity
  Matrix3cd testLink = lattice.getLink(linkCoords);
  BOOST_CHECK(areEqual(testLink * testLink.adjoint(), Matrix3cd::Identity(),
		       100 * DBL_EPSILON));
  BOOST_CHECK(areEqual(testLink.determinant(), 1.0, 100 * DBL_EPSILON));

  // Checking parallel Monte Carlo updates
  lattice = Lattice(8, 8, 5.5, 1.0, 0, 60, 1, 1, 4);
  lattice.thermalize();
  // Check basic observables
  BOOST_CHECK(lattice.computeAveragePlaquette() < 0.51 &&
	      lattice.computeAveragePlaquette() > 0.49);
  BOOST_CHECK(lattice.computeAverageRectangle() < 0.27 &&
	      lattice.computeAverageRectangle() > 0.25);
  // Check unitarity
  testLink = lattice.getLink(linkCoords);
  BOOST_CHECK(areEqual(testLink * testLink.adjoint(), Matrix3cd::Identity(),
		       100 * DBL_EPSILON));
  BOOST_CHECK(areEqual(testLink.determinant(), 1.0, 100 * DBL_EPSILON));

  // Checking serial heatbath
  lattice = Lattice(8, 8, 5.5, 1.0, 0, 10, 0, 0, 4);
  lattice.thermalize();
  // Check basic observables
  BOOST_CHECK(lattice.computeAveragePlaquette() < 0.51 &&
	      lattice.computeAveragePlaquette() > 0.49);
  BOOST_CHECK(lattice.computeAverageRectangle() < 0.27 &&
	      lattice.computeAverageRectangle() > 0.25);
  // Check unitarity
  testLink = lattice.getLink(linkCoords);
  BOOST_CHECK(areEqual(testLink * testLink.adjoint(), Matrix3cd::Identity(),
		       100 * DBL_EPSILON));
  BOOST_CHECK(areEqual(testLink.determinant(), 1.0, 100 * DBL_EPSILON));

  // Checking serial Monte Carlo
  lattice = Lattice(8, 8, 5.5, 1.0, 0, 60, 1, 0, 4);
  lattice.thermalize();
  // Check basic observables
  BOOST_CHECK(lattice.computeAveragePlaquette() < 0.51 &&
	      lattice.computeAveragePlaquette() > 0.49);
  BOOST_CHECK(lattice.computeAverageRectangle() < 0.27 &&
	      lattice.computeAverageRectangle() > 0.25);
  // Check unitarity
  testLink = lattice.getLink(linkCoords);
  BOOST_CHECK(areEqual(testLink * testLink.adjoint(), Matrix3cd::Identity(),
		       100 * DBL_EPSILON));
  BOOST_CHECK(areEqual(testLink.determinant(), 1.0, 100 * DBL_EPSILON));
}
