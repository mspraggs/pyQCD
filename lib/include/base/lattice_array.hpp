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
    LatticeArray(const std::vector<int>& lattice_shape,
		 const std::vector<int>& block_shape
		 = std::vector<int>(NDIM, 2));
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

    // Utility functions specific to the lattice layout
    std::vector<int> get_site_coords(const int index) const;
    template <typename U>
    void get_site_coords(const int index, U& site_coords) const;
    template <typename U>
    void get_site_index(const U& site_coords) const;

  protected:
    // The data we're wrapping. We use a vector of vectors to
    // implement some sort of cache blocking: the lattice is
    // sub-divided into blocks to reduce cache misses by
    // improving locality.
    std::vector<std::vector<T> > _data;

  private:
    std::vector<int> _lattice_shape;
    std::vector<int> _block_shape;
    std::vector<std::vector<int> > _layout;
    int _lattice_volume;
    int _num_blocks;
    int _block_volume;
  };
}

#endif
