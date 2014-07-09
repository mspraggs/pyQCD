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
#include <exception>
#include <string>
#include <numeric>
#include <functional>
#include <algorithm>

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
    const T& operator()(COORD_INDEX_ARGS(n)) const;
    T& operator()(COORD_INDEX_ARGS(n));

    // Functions to access the _data member directly
    T& operator[](const int index);
    const T& operator[](const int index) const;
    T& datum_ref(const int i, const int j);
    const T& datum_ref(const int i, const int j) const;

    // Utility functions specific to the lattice layout
    std::vector<int> get_site_coords(const int index) const;
    template <typename U>
    void get_site_coords(const int index, U& site_coords) const;
    template <typename U>
    int get_site_index(const U& site_coords) const;

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


  
  template <typename T>
  LatticeArray<T>::LatticeArray(const std::vector<int>& lattice_shape,
				const std::vector<int>& block_shape)
    : _lattice_shape(lattice_shape), _block_shape(block_shape)
  {
    // First sanity check the input
    if (lattice_shape.size() == NDIM || block_shape.size() != NDIM) {
      // Then check that the blocks can fit inside the lattice
      for (int i = 0; i < NDIM; ++i)
	if (lattice_shape[i] % block_shape[i] != 0) {
	  std::string msg = "Lattice shape is not integer multiple "
	    "of block shape along dimension ";
	  msg += std::to_string(i);
	  throw std::invalid_argument(msg);
	}

      // Now compute the total number of sites
      this->_lattice_volume = std::accumulate(lattice_shape.begin(),
					      lattice_shape.end(), 1,
					      std::multiplies<int>());
      // And the block volume
      this->_block_volume = std::accumulate(block_shape.begin(),
					    block_shape.end(), 1,
					    std::multiplies<int>());

      this->_num_blocks = this->_lattice_volume / this->_block_volume;
      // Resize the _data vector
      this->_data.resize(this->_num_blocks);
      this->_layout.resize(this->_lattice_volume);

      // Now that we have the number of sites, iterate through
      // the lexicographic indices of the sites, compute the
      // coordinates, then assign a block and block site
      std::vector<int> coords(NDIM, 0);
      for (int i = 0; i < this->_num_sites; ++i) {
	// Resize the current _layout sub-vector
	this->_layout[i] = std::vector<int>(2, 0);
	// Determine the coordinates of the current site
	this->get_site_coords(i, coords);

	// Now determine the coordinates of the current block relative to the
	// lattice and the coordinates of the current site relative to the block
	// Block relative to lattice:
	std::vector<int> lattice_block_coords(NDIM, 0);
	std::transform(coords.begin(), coords.end(), block_shape.begin(),
		       lattice_block_coords.begin(), std::divides<int>());
	// Site relative to block:
	std::vector<int> block_site_coords(NDIM, 0);
	std::transform(coords.begin(), coords.end(), block_shape.begin(),
		       block_site_coords.begin(), std::modulus<int>());

	// Now determine the lexicographical index of the block within
	// the lattice and the lexicographical index of the site within
	// the block.
	int lattice_block_index = 0;
	int block_site_index = 0;
	for (int j = 0; j < NDIM; ++j) {
	  lattice_block_index *= lattice_shape[j] / block_shape[j];
	  lattice_block_index += lattice_block_coords[j];

	  block_site_index *= block_shape[j];
	  block_site_index += block_site_coords[j];
	}

	// Determine if the site will change the half of the block it resides
	// in due to even-odd ordering. Because we're about to divide the
	// lexicographic indices by two (to account for even-odd ordering), we
	// need some form of compensation factor to prevent some sites from
	// ending up in the same block and block site. Here, we add half a block
	// volume whenever the block lexicographic index is odd.
	int block_site_index_shift
	  = (lattice_block_index % 2 > 0) ? this->_block_volume / 2 : 0;

	// Since we're doing everything even-odd stylee, we divide the two
	// computed indices by two.
	block_site_index /= 2;
	lattice_block_index /= 2;

	// In much the same way as within the block, we need a correction factor
	// for the fact that we've just divided the lattice_block_index by 2
	// Here, odd sites get moved to the second set of blocks.
	if (std::accumulate(coords.begin(), coords.end(), 0) % 2 > 0)
	  lattice_block_index += this->_num_blocks / 2;

	// Assign those blocks
	this->_layout[i][0] = lattice_block_index;
	this->_layout[i][1] = block_site_index;
      } // End loop over sites
      
      // Now we've configured the layout, we proceed to initialize the variables
      // within _data

      for (std::vector<T>& inner : this->_data)
	for (T& datum: inner)
	  datum = T();
    }
    else {
      // The input was bad (lattice shape exceeds number of dimesnions)
      // so throw an error
      std::string msg = "Lattice or block shape does not have dimension ";
      msg += std::to_string(NDIM);
      throw std::invalid_argument(msg);
    }
  }



  template <typename T>
  LatticeArray<T>::LatticeArray(const LatticeArray& lattice_array)
    : _data(lattice_array._data),
      _lattice_shape(lattice_array._lattice_shape),
      _block_shape(lattice_array._block_shape),
      _layout(lattice_array._layout),
      _lattice_volume(lattice_array._lattice_volume),
      _num_blocks(lattice_array._num_blocks),
      _block_volume(lattice_array._block_volume)
  {
    // Copy constructor, we'll all done here
  }



  template <typename T>
  LatticeArray<T>::~LatticeArray()
  {
    // Destructor - nothing to do here
  }



  template <typename T>
  LatticeArray<T>& LatticeArray<T>::operator=(const LatticeArray<T>& rhs)
  {
    if (this != &rhs) {
      this->_data = rhs._data;
      this->_lattice_shape = rhs._lattice_shape;
      this->_lattice_shape = rhs._block_shape;
      this->_layout = rhs._layout;
      this->_lattice_volume = rhs._lattice_volume;
      this->_num_blocks = rhs._num_blocks;
      this->_block_volume = rhs._block_volume;
    }

    return *this;
  }
}

#endif
