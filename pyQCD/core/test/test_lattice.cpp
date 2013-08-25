#define BOOST_TEST_MODULE Lattice test
#include <lattice.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <Eigen/Dense>
#include <complex>
#include <cstdlib>

using namespace Eigen;
using namespace std;

class exposedLattice: public Lattice
{
public:
  exposedLattice(const int spatialExtent = 4,
		 const int temporalExtent = 8,
		 const double beta = 5.5,
		 const double u0 = 1.0,
		 const int action = 0,
		 const int nCorrelations = 50,
		 const int updateMethod = 0,
		 const int parallelFlag = 1,
		 const int chunkSize = 4,
		 const int randSeed = -1) :
    Lattice::Lattice(spatialExtent, temporalExtent, beta, u0, action,
		     nCorrelations, updateMethod, parallelFlag, chunkSize,
		     randSeed)
  {
    
  }

  ~exposedLattice()
  {
    
  }

  
  double computeLocalWilsonAction(const int link[5])
  {
    return Lattice::computeLocalWilsonAction(link);
  }

  double computeLocalRectangleAction(const int link[5])
  {
    return Lattice::computeLocalRectangleAction(link);
  }

  double computeLocalTwistedRectangleAction(const int link[5])
  {
    return Lattice::computeLocalTwistedRectangleAction(link);
  }

  Matrix3cd computeWilsonStaples(const int link[5])
  {
    return Lattice::computeWilsonStaples(link);
  }

  Matrix3cd computeRectangleStaples(const int link[5])
  {
    return Lattice::computeRectangleStaples(link);
  }
};

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
      BOOST_CHECK_CLOSE(lattice.computePlaquette(site, i, j),
			plaquetteVal, 1e-11);
      BOOST_CHECK_CLOSE(lattice.computeRectangle(site, i, j),
			rectangleVal, 1e-11);
      BOOST_CHECK_CLOSE(lattice.computeTwistedRectangle(site, i, j),
			twistedRectangleVal, 1e-11);
    }
  }

  double nSites = 6 * lattice.temporalExtent * pow(lattice.spatialExtent, 3);
  // Checking average plaquette and rectangle
  BOOST_CHECK_CLOSE(lattice.computeAveragePlaquette(),
		    plaquetteVal, 1e-11 * nSites);
  BOOST_CHECK_CLOSE(lattice.computeAverageRectangle(),
		    rectangleVal, 1e-11 * nSites);
  // Checking average Wilson loops
  BOOST_CHECK_CLOSE(lattice.computeAverageWilsonLoop(1, 1),
		    plaquetteVal, 1e-11 * nSites);
  BOOST_CHECK_CLOSE(lattice.computeAverageWilsonLoop(2, 2),
		    twistedRectangleVal, 1e-11 * nSites);

  // Check the link smearing
  exposedLattice nonRandomLattice(4, 8, 5.5, 1.0, 0, 10, 0, 0, 4, 0);
  nonRandomLattice.update();
  nonRandomLattice.smearLinks(0, 1, 0.5);

  double realComponents[3] = {0.89851667094939247082,
			      0.40553008395662609731,
			      0.40788086130924816608};

  double imagComponents[3] = {-0.21221110562333689309,
			      0.732592736591202498,
			      -0.018348970642218388749};

  for (int i = 1; i < 4; ++i) {
    int link[5] = {0, 0, 0, 0, i};
    BOOST_CHECK_CLOSE(nonRandomLattice.getLink(link)(0, 0).real(),
		      realComponents[i - 1], 1e-11);
    BOOST_CHECK_CLOSE(nonRandomLattice.getLink(link)(0, 0).imag(),
		      imagComponents[i - 1], 1e-11);
  }
}

BOOST_AUTO_TEST_CASE( action_test )
{
  exposedLattice lattice;
  // Initialise the lattice with a randome complex matrix
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
  
  // Calculate plaquette, rectangle and twisted rectangle values
  double plaquetteVal = pow(abs(randC), 4);
  double rectangleVal = pow(abs(randC), 6);
  double twistedRectangleVal = pow(abs(randC), 8);
  // Use these values to calculate the actions
  double wilsonAction = -6 * plaquetteVal;
  double rectangleAction = 5.0 / 3.0 * wilsonAction + 18 / 12.0 * rectangleVal;
  double twistedRectangleAction = wilsonAction - 21 / 12.0 * twistedRectangleVal;

  // Compare the actions as calculated on the lattice with those calculated
  // above
  int link[5] = {0, 0, 0, 0, 0};
  BOOST_CHECK_CLOSE(lattice.computeLocalWilsonAction(link),
		    5.5 * wilsonAction, 1e-11);
  BOOST_CHECK_CLOSE(lattice.computeLocalRectangleAction(link),
		    5.5 * rectangleAction, 1e-11);
  BOOST_CHECK_CLOSE(lattice.computeLocalTwistedRectangleAction(link),
		    5.5 * twistedRectangleAction, 1e-11);

  // Calculate the staples
  Matrix3cd wilsonStaples = lattice.computeWilsonStaples(link);
  Matrix3cd rectangleStaples = lattice.computeRectangleStaples(link);
  Matrix3cd linkMatrix = lattice.getLink(link);
  // Compare the lattice staples with the calculated action
  BOOST_CHECK_CLOSE(1.0 / 3.0 * (linkMatrix * wilsonStaples).trace().real(),
		    -wilsonAction, 1e-11);
  BOOST_CHECK_CLOSE(1.0 / 3.0 * (linkMatrix * rectangleStaples)
		    .trace().real(), -rectangleAction,
		    1e-11);
}

BOOST_AUTO_TEST_CASE( update_test )
{
  int linkCoords[5] = {0, 0, 0, 0, 0};

  // Checking heatbath updates
  exposedLattice lattice(4, 8, 5.5, 1.0, 0, 10, 0, 0, 4, 0);

  // First check they preserver unitarity - do a single update
  lattice.heatbath(0);
  BOOST_CHECK(areEqual(lattice.getLink(linkCoords)
		       * lattice.getLink(linkCoords).adjoint(),
		       Matrix3cd::Identity(),
		       1e-11));
  BOOST_CHECK_CLOSE(lattice.getLink(linkCoords).determinant().real(), 1.0,
		    1e-11);
  BOOST_CHECK_SMALL(lattice.getLink(linkCoords).determinant().imag(),
		    100 * DBL_EPSILON);
  BOOST_CHECK_CLOSE(lattice.computePlaquette(linkCoords, 0, 1),
		    0.93548039868213483938, 1e-11);

  // Now check the Metropolis updates
  lattice = exposedLattice(4, 8, 5.5, 1.0, 0, 10, 1, 0, 4, 0);
  // Do a single update
  lattice.metropolis(0);
  // Check unitarity and expected plaquette value
  BOOST_CHECK(areEqual(lattice.getLink(linkCoords)
		       * lattice.getLink(linkCoords).adjoint(),
		       Matrix3cd::Identity(),
		       1e-11));
  BOOST_CHECK_CLOSE(lattice.getLink(linkCoords).determinant().real(), 1.0,
		    1e-11);
  BOOST_CHECK_SMALL(lattice.getLink(linkCoords).determinant().imag(),
		    100 * DBL_EPSILON);
  BOOST_CHECK_CLOSE(lattice.computePlaquette(linkCoords, 0, 1),
		    0.98480251715017408376, 1e-11);

  // Now check that about 50% of the updates are accepted.
  double firstAction = lattice.computeLocalWilsonAction(linkCoords);
  double oldAction = firstAction;
  int numDecreases = 0;
  // Do 100 metropolis updates. Should expect numDecreases ~ 50
  for (int i = 0; i < 100; ++i) {
    lattice.metropolis(0);
    if (lattice.computeLocalWilsonAction(linkCoords) < oldAction)
      numDecreases++;
    oldAction = lattice.computeLocalWilsonAction(linkCoords);
  }

  // Metropolis should change the action roughly 50% of the time.
  BOOST_CHECK_CLOSE(double(numDecreases), 50.0, 10.0);
}
