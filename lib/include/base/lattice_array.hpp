/* This file provides a container for lattice-wide objects. This class serves as
 * a base for all other lattice objects, e.g. LatticeGaugeField, LatticeSpinor,
 * etc.
 *
 * The container handles the memory layout for these types, hopefully in a way
 * that reduces cache misses by blocking neighbouring sites together. All even
 * sites are blocked together, and all odd sites are blocked together, since
 * Dirac operators and so on often require access to only one type of site.
 *
 * This file also contains expression templates to hopefully optimize any
 * arithmetic operations involving LatticeArray.
 */

template <typename T>
class LatticeArray
{

public:
  LatticeArray();

protected:
  vector<vector<T> > data_;

private:
  vector<vector<int> > layout_;
};
