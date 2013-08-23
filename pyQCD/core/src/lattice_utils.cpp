#include <lattice.hpp>
#include <utils.hpp>

void Lattice::print()
{
  // Print the links out. A bit redundant due to the interfaces library,
  // but here in case it's needed.
  for (int i = 0; i < this->nLinks_; ++i) {
    cout << this->links_[i] << endl;
  }
}



Matrix3cd& Lattice::getLink(const int link[5])
{
  // Return link specified by index (sanitizes link indices)
  int tempLink[5];
  tempLink[0] = pyQCD::mod(link[0], this->temporalExtent);
  for (int i = 1; i < 4; ++i)
    tempLink[i] = pyQCD::mod(link[i], this->spatialExtent);
  tempLink[4] = pyQCD::mod(link[4], 4);

  int index = pyQCD::getLinkIndex(tempLink, this->spatialExtent);
  
  return this->links_[index];
}



Matrix3cd& Lattice::getLink(const vector<int> link)
{
  // Return link specified by indices
  int tempLink[5];
  tempLink[0] = pyQCD::mod(link[0], this->temporalExtent);
  for (int i = 1; i < 4; ++i)
    tempLink[i] = pyQCD::mod(link[i], this->spatialExtent);
  tempLink[4] = pyQCD::mod(link[4], 4);

  int index = pyQCD::getLinkIndex(tempLink, this->spatialExtent);

  return this->links_[index];
}



void Lattice::setLink(const int link[5], const Matrix3cd& matrix)
{
  // Set the value of a link
  this->getLink(link) = matrix;
}



GaugeField Lattice::getSubLattice(const int startIndex, const int size)
{
  // Returns a GaugeField object corresponding to the sub-lattice starting at
  // link index startIndex

  GaugeField out;
  out.resize(size * size * size * size * 4);

  int incrementOne = 4 * this->spatialExtent;
  int incrementTwo = incrementOne * this->spatialExtent;
  int incrementThree = incrementTwo * this->spatialExtent;
  
  int index = 0;

  for (int i = 0; i < size * incrementThree; i += incrementThree) {
    for (int j = 0; j < size * incrementTwo; j += incrementTwo) {
      for (int k = 0; k < size * incrementOne; k += incrementOne) {
	for (int l = 0; l < 4 * size; ++l) {
	  out[index] = this->links_[i + j + k + l];
	  ++index;
	}
      }
    }
  }

  return out;
}
