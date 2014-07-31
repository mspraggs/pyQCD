#ifndef LATTICE_BASE_HPP
#define LATTICE_BASE_HPP

/* This file provides a container for lattice-wide objects. This class serves as
 * a base for all other lattice objects, e.g. LatticeGaugeField, LatticeSpinor,
 * etc.
 *
 * The container handles the memory layout for these types, hopefully in a way
 * that reduces cache misses by blocking neighbouring sites together. All even
 * sites are blocked together, and all odd sites are blocked together, since
 * Dirac operators and so on often require access to only one type of site.
 *
 * The expression templates to optimize the arithmetic for this class can be
 * found in lattice_base_expr.hpp.
 */

#include <vector>
#include <exception>
#include <string>
#include <numeric>
#include <functional>
#include <algorithm>
#include <stdexcept>

#include <utils/math.hpp>

#include <base/lattice_base_expr.hpp>

namespace pyQCD
{
  template <typename T, int ndim>
  class LatticeBase : public LatticeBaseExpr<LatticeBase<T, ndim>, T>
  {
    template <typename X, typename Y>
    friend class LatticeBaseExpr;
    template <typename X, typename Y>
    friend class LatticeBaseEven;
    template <typename X, typename Y, typename Z>
    friend class LatticeBaseSum;
    template <typename X, typename Y, typename Z>
    friend class LatticeBaseDiff;
    template <typename X, typename Y, typename Z>
    friend class LatticeBaseMult;
    template <typename X, typename Y, typename Z>
    friend class LatticeBaseDiv;
  public:
    // Constructors
    LatticeBase(const std::vector<int>& lattice_shape,
		const std::vector<int>& block_shape
		= std::vector<int>(ndim, 4));
    LatticeBase(const T& init_value,
		const std::vector<int>& lattice_shape,
		const std::vector<int>& block_shape
		= std::vector<int>(ndim, 4));
    LatticeBase(const LatticeBase<T, ndim>& lattice_base);
    template <typename U>
    LatticeBase(const LatticeBaseExpr<U, T>& expr)
    {
      // Copy constructor to create a LatticeBase from its respective
      // expression template
      const U& e = expr;
      this->_lattice_shape = e.lattice_shape();
      this->_block_shape = e.block_shape();
      this->_layout = e.layout();
      this->_lattice_volume = e.lattice_volume();
      this->_num_blocks = e.num_blocks();
      this->_block_volume = e.block_volume();
      this->_data.resize(e.num_blocks());
      for (int i = 0; i < e.num_blocks(); ++i) {
	this->_data[i].resize(e.block_volume());
	for (int j = 0; j < e.block_volume(); ++j)
	  this->_data[i][j] = e.datum_ref(i, j);
      }
    }
    virtual ~LatticeBase() { };

    // Equality operator overload
    LatticeBase<T, ndim>& operator=(const LatticeBase<T, ndim>& rhs);
    LatticeBase<T, ndim>& operator=(const T& rhs);

    // Functions to access the _data member directly
    T& operator[](const int index);
    const T& operator[](const int index) const;
    template <typename U>
    T& operator[](const U& coords);
    template <typename U>
    const T& operator[](const U& coords) const;
    T& datum_ref(const int i, const int j);
    const T& datum_ref(const int i, const int j) const;

    // Arithmetic operator overloads
    template <typename U>
    LatticeBase<T, ndim>& operator*=(const U& scalar);
    template <typename U>
    LatticeBase<T, ndim>& operator/=(const U& scalar);
    LatticeBase<T, ndim>& operator+=(const LatticeBase<T, ndim>& rhs);
    LatticeBase<T, ndim>& operator-=(const LatticeBase<T, ndim>& rhs);

    // Utility functions specific to the lattice layout
    std::vector<int> get_site_coords(const int index) const;
    template <typename U>
    void get_site_coords(const int index, U& site_coords) const;
    template <typename U>
    int get_site_index(const U& site_coords) const;

    // Member access functions
    const std::vector<int>& lattice_shape() const
    { return this->_lattice_shape; }
    const int lattice_volume() const
    { return this->_lattice_volume; }

    LatticeBaseEven<LatticeBase<T, ndim>, T> even_sites()
    { return LatticeBaseEven<LatticeBase<T, ndim>, T>(*this); }

  protected:
    // The data we're wrapping. We use a vector of vectors to
    // implement some sort of cache blocking: the lattice is
    // sub-divided into blocks to reduce cache misses by
    // improving locality.
    std::vector<std::vector<T> > _data;
    // The shape of the lattice
    std::vector<int> _lattice_shape;
    // The shape of the blocks used for cache blocking
    std::vector<int> _block_shape;
    // The layout of the sites within the lattice. The index for the outer
    // vector corresponds to the lexicographic index of a site relative to
    // the origin of the entire lattice. The inner vector is of length two.
    // The first element is the lexicographic index of the block (in terms
    // of the number of blocks) relative to the origin of the lattice. The
    // second element is the lexicographic index of the site relative to
    // the origin of the block in which it resides.
    std::vector<std::vector<int> > _layout;
    int _lattice_volume; // Total number of lattice sites
    int _num_blocks; // Total number of blocks
    int _block_volume; // The number of sites in each block.

  private:
    // Common constructor code
    void init(const std::vector<int>& lattice_shape,
              const std::vector<int>& block_shape);

    const std::vector<int>& block_shape() const
    { return this->_block_shape; }
    const std::vector<std::vector<int> >& layout() const
    { return this->_layout; }
    const int num_blocks() const
    { return this->_num_blocks; }
    const int block_volume() const
    { return this->_block_volume; }
  };


  
  template <typename T, int ndim>
  LatticeBase<T, ndim>::LatticeBase(const std::vector<int>& lattice_shape,
				    const std::vector<int>& block_shape)
    : _lattice_shape(lattice_shape), _block_shape(block_shape)
  {
    // Constructor for given lattice size etc -> values in _data initialized
    // to default for type T
    this->init(lattice_shape, block_shape);
    
    // Now we've configured the layout, we proceed to initialize the variables
    // within _data    
    for (std::vector<T>& inner : this->_data)
      for (T& datum: inner)
	datum = T();
  }


  
  template <typename T, int ndim>
  LatticeBase<T, ndim>::LatticeBase(const T& init_value,
				    const std::vector<int>& lattice_shape,
				    const std::vector<int>& block_shape)
    : _lattice_shape(lattice_shape), _block_shape(block_shape)
  {
    // Constructor for given lattice size etc -> values in _data initialized
    // as all equal to the specified value.
    this->init(lattice_shape, block_shape);
    
    // Now we've configured the layout, we proceed to initialize the variables
    // within _data
    
    for (std::vector<T>& inner : this->_data)
      for (T& datum: inner)
	datum = init_value;
  }



  template <typename T, int ndim>
  LatticeBase<T, ndim>::LatticeBase(const LatticeBase& lattice_base)
    : _data(lattice_base._data),
      _lattice_shape(lattice_base._lattice_shape),
      _block_shape(lattice_base._block_shape),
      _layout(lattice_base._layout),
      _lattice_volume(lattice_base._lattice_volume),
      _num_blocks(lattice_base._num_blocks),
      _block_volume(lattice_base._block_volume)
  {
    // Copy constructor, we'll all done here
  }



  template <typename T, int ndim>
  void LatticeBase<T, ndim>::init(const std::vector<int>& lattice_shape,
			     const std::vector<int>& block_shape)
  {
    // Common include code - determines lattice site layout for the given
    // lattice_shape and block_shape

    // First sanity check the input
    if (lattice_shape.size() == ndim || block_shape.size() != ndim) {
      // Then check that the blocks can fit inside the lattice
      for (int i = 0; i < ndim; ++i)
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
      for (std::vector<T>& inner : this->_data)
	inner.resize(this->_block_volume);
      this->_layout.resize(this->_lattice_volume);

      // Now that we have the number of sites, iterate through
      // the lexicographic indices of the sites, compute the
      // coordinates, then assign a block and block site
      std::vector<int> coords(ndim, 0);
      for (int i = 0; i < this->_lattice_volume; ++i) {
	// Resize the current _layout sub-vector
	this->_layout[i] = std::vector<int>(2, 0);
	// Determine the coordinates of the current site
	this->get_site_coords(i, coords);

	// Now determine the coordinates of the current block relative to the
	// lattice and the coordinates of the current site relative to the block
	// Block relative to lattice:
	std::vector<int> lattice_block_coords(ndim, 0);
	std::transform(coords.begin(), coords.end(), block_shape.begin(),
		       lattice_block_coords.begin(), std::divides<int>());
	// Site relative to block:
	std::vector<int> block_site_coords(ndim, 0);
	std::transform(coords.begin(), coords.end(), block_shape.begin(),
		       block_site_coords.begin(), std::modulus<int>());

	// Now determine the lexicographical index of the block within
	// the lattice and the lexicographical index of the site within
	// the block.
	int lattice_block_index = 0;
	int block_site_index = 0;
	for (int j = 0; j < ndim; ++j) {
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
	this->_layout[i][1] = block_site_index + block_site_index_shift;
      } // End loop over sites
    }
    else {
      // The input was bad (lattice shape exceeds number of dimesnions)
      // so throw an error
      std::string msg = "Lattice or block shape does not have dimension ";
      msg += std::to_string(ndim);
      throw std::invalid_argument(msg);
    }
  }



  template <typename T, int ndim>
  LatticeBase<T, ndim>&
  LatticeBase<T, ndim>::operator=(const LatticeBase<T, ndim>& rhs)
  {
    if (this != &rhs) {
      this->_data = rhs._data;
      this->_lattice_shape = rhs._lattice_shape;
      this->_block_shape = rhs._block_shape;
      this->_layout = rhs._layout;
      this->_lattice_volume = rhs._lattice_volume;
      this->_num_blocks = rhs._num_blocks;
      this->_block_volume = rhs._block_volume;
    }

    return *this;
  }



  template <typename T, int ndim>
  LatticeBase<T, ndim>& LatticeBase<T, ndim>::operator=(const T& rhs)
  {
    // Set all site values to the supplied constant
    if (this->_data.size() == 0 && this->_data[0].size() == 0)
      throw std::out_of_range("Assigning constant value to empty LatticeBase");
    for (auto& block : this->_data)
      for (auto& site : block)
	site = rhs;      

    return *this;
  }



  template <typename T, int ndim>
  const T& LatticeBase<T, ndim>::operator[](const int index) const
  {
    // Returns a constant reference to the element in _datum specified by the
    // given lexicographic lattice site index.
    return this->_data[this->_layout[index][0]][this->_layout[index][1]];
  }



  template <typename T, int ndim>
  T& LatticeBase<T, ndim>::operator[](const int index)
  {
    // Returns a reference to the element in _datum specified by the given 
    // lexicographic lattice site index.
    return this->_data[this->_layout[index][0]][this->_layout[index][1]];
  }


  
  template <typename T, int ndim>
  template <typename U> 
  const T& LatticeBase<T, ndim>::operator[](const U& coords) const
  {
    // Returns a constant reference to the object at the lattice site specified
    // by the integer coordinates in the supplied vector
    int site_index = this->get_site_index(coords);
    return (*this)[site_index];
  }



  template <typename T, int ndim>
  template <typename U>
  T& LatticeBase<T, ndim>::operator[](const U& coords)
  {
    // Returns a reference to the object at the lattice site specified
    // by the integer coordinates in the supplied vector
    int site_index = this->get_site_index(coords);
    return (*this)[site_index];
  }



  template <typename T, int ndim>
  const T& LatticeBase<T, ndim>::datum_ref(const int i, const int j) const
  {
    // Returns a constant reference to the element in _datum specified by the
    // given vector indices i and j
    return this->_data[i][j];
  }



  template <typename T, int ndim>
  T& LatticeBase<T, ndim>::datum_ref(const int i, const int j)
  {
    // Returns a reference to the element in _datum specified by the
    // given vector indices i and j
    return this->_data[i][j];
  }



  template <typename T, int ndim>
  template <typename U>
  LatticeBase<T, ndim>& LatticeBase<T, ndim>::operator*=(const U& scalar)
  {
    // Multiply whole LatticeBase by a scalar value
    for (std::vector<T>& inner : this->_data)
      for (T& datum : inner)
	datum *= scalar;
    return *this;
  }



  template <typename T, int ndim>
  template <typename U>
  LatticeBase<T, ndim>& LatticeBase<T, ndim>::operator/=(const U& scalar)
  {
    // Multiply whole LatticeBase by a scalar value
    for (std::vector<T>& inner : this->_data)
      for (T& datum : inner)
	datum /= scalar;
    return *this;
  }



  template <typename T, int ndim>
  LatticeBase<T, ndim>&
  LatticeBase<T, ndim>::operator+=(const LatticeBase<T, ndim>& rhs)
  {
    // Add the given LatticeBase to this LatticeBase
    for (int i = 0; i < this->_num_blocks; ++i)
      for (int j = 0; j < this->_block_volume; ++j)
	this->_data[i][j] += rhs._data[i][j];
    return *this;
  }



  template <typename T, int ndim>
  LatticeBase<T, ndim>&
  LatticeBase<T, ndim>::operator-=(const LatticeBase<T, ndim>& rhs)
  {
    // Add the given LatticeBase to this LatticeBase
    for (int i = 0; i < this->_num_blocks; ++i)
      for (int j = 0; j < this->_block_volume; ++j)
	this->_data[i][j] -= rhs._data[i][j];
    return *this;
  }



  template <typename T, int ndim>
  std::vector<int> LatticeBase<T, ndim>::get_site_coords(const int index) const
  {
    // Computes the coordinates of a site from the specified lexicographic index
    std::vector<int> out(ndim, 0);
    this->get_site_coords(index, out);
    return out;
  }



  template <typename T, int ndim>
  template <typename U>
  void LatticeBase<T, ndim>::get_site_coords(const int index,
					     U& site_coords) const
  {
    // Gets the coordinates of the site with the specified lexicographic index
    int temp_index = index;
    for (int i = ndim - 1; i >= 0; --i) {
      // Here we're basically doing the reverse of get_site_index
      site_coords[i] = temp_index % this->_lattice_shape[i];
      temp_index /= this->_lattice_shape[i];
    }   
  }

  

  template <typename T, int ndim>
  template <typename U>
  int LatticeBase<T, ndim>::get_site_index(const U& site_coords) const
  {
    // Computes the lexicographic site index from the specified site
    // coordinates.
    // We're basically coming up with an index computed as follows:
    // index = x_n + N_1 * (x_{n-1} + ... (x_1 + N_{n-1} * x_0) ... )
    int index = 0;
    for (int i = 0; i < ndim; ++i) {
      index *= this->_lattice_shape[i];
      index += mod(site_coords[i], this->_lattice_shape[i]);
    }
    return index;
  }
}

#endif
