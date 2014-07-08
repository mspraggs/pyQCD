#ifndef LATTICE_ARRAY_HPP
#define LATTICE_ARRAY_HPP

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

#include <vector>

#include <utils/macros.hpp>

namespace pyQCD
{
  template <typename T>
  class LatticeArray
  {

  public:
    // Constructors
    LatticeArray(const vector<int>& lattice_shape);
    LatticeArray(const LatticeArray<T>& lattice_array);
    virtual ~LatticeArray();

    // Operator overloads
    LatticeArray<T>& operator=(const LatticeArray<T>& rhs);
    const T& operator[](COORD_INDEX_ARGS(n)) const;
    T& operator[](COORD_INDEX_ARGS(n));

    // Functions to access the _data member directly
    T& datum_ref(const int index);
    const &T datum_ref(const int index) const;
    T& datum_ref(const int i, const int j);
    const &T datum_ref(const int i, const int j) const;

  protected:
    vector<vector<T> > _data;

  private:
    vector<int> _shape;
    vector<vector<int> > _layout;
    int _num_sites;
  };
}

#endif
