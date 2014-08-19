#ifndef LATTICE_BASE_EXP_HPP
#define LATTICE_BASE_EXP_HPP

/* This file provides expression templates for the LatticeBase class, so that
 * temporaries do not need to be created when performing arithmetic operations.
 *
 * The idea is to use template metaprogramming to represent arithmetic
 * operations at compile time so that the expression can be compiled down to
 * expressions that iterate over the elements of the LatticeBase, thus negating
 * the need for allocating temporaries.
 */

#include <numeric>
#include <functional>



namespace pyQCD
{
  template <typename T, int ndim>
  class LatticeBase;

  template <typename T, typename U>
  class LatticeBaseExpr;

  template <typename T, typename U, bool even>
  class LatticeBaseSubset;

  template <typename T, typename U>
  class LatticeBaseConst;


  template <typename T, typename U>
  class SubsetTraits
  {
  public:
    typedef const T& member_type;
    typedef const LatticeBaseExpr<T, U>& constructor_type;
  };



  template <typename T, int ndim>
  class SubsetTraits<LatticeBase<T, ndim>, T>
  {
  public:
    typedef LatticeBase<T, ndim>& member_type;
    typedef LatticeBaseExpr<LatticeBase<T, ndim>, T>& constructor_type;
  };



  template <typename T>
  class BinaryTraits
  {
  public:
    typedef const T& type;
  };



  template <typename T, typename U>
  class BinaryTraits<LatticeBaseConst<T, U> >
  {
  public:
    typedef LatticeBaseConst<T, U> type;
  };



  template <typename T, typename U>
  class LatticeBaseExpr
  {
    // This is the main expression class from which all others are derived. It
    // It takes two template arguments: the type that's involved in expressions

    // and the site type used in LatticeBase. This allows expressions to be
    // abstracted to a nested hierarchy of types. When the compiler goes through
    // and does it's thing, the definitions of the operations within these
    // template classes are all spliced together.
  public:
    // Here we use curiously recurring template patterns to call functions in
    // the LatticeBase type.
    U& datum_ref(const int i, const int j)
    { return static_cast<T&>(*this).datum_ref(i, j); }
    const U& datum_ref(const int i, const int j) const
    { return static_cast<const T&>(*this).datum_ref(i, j); }

    // Functions to grab the size of the lattice, layout etc.
    const std::vector<int>& lattice_shape() const
    { return static_cast<const T&>(*this).lattice_shape(); }
    const std::vector<int>& block_shape() const
    { return static_cast<const T&>(*this).block_shape(); }
    const std::vector<std::vector<int> >& layout() const
    { return static_cast<const T&>(*this).layout(); }
    const int lattice_volume() const
    { return static_cast<const T&>(*this).lattice_volume(); }
    const int num_blocks() const
    { return static_cast<const T&>(*this).num_blocks(); }
    const int block_volume() const
    { return static_cast<const T&>(*this).block_volume(); }

    const LatticeBaseSubset<T, U, true> even_sites() const
    { return LatticeBaseSubset<T, U, true>(*this); }
    const LatticeBaseSubset<T, U, false> odd_sites() const
    { return LatticeBaseSubset<T, U, false>(*this); }

    operator T&()
    { return static_cast<T&>(*this); }
    operator T const&() const
    { return static_cast<const T&>(*this); }
  };



  template <typename T, typename U, bool even>
  class LatticeBaseSubset
    : public LatticeBaseExpr<LatticeBaseSubset<T, U, even>, U>
  {
    // Generates a reference to the even sites in a lattice
  public:
    // Constructed from a reference to a LatticeBaseExpr.
    LatticeBaseSubset(typename SubsetTraits<T, U>::constructor_type lattice)
      : _lattice(lattice)
    { }

    LatticeBaseSubset& operator=(const LatticeBaseSubset<T, U, even>& rhs)
    {
      typedef LatticeBaseExpr<LatticeBaseSubset<T, U, even>, U> cast_type;
      return operator=(static_cast<const cast_type&>(rhs));
    }

    template <typename V>
    LatticeBaseSubset& operator=(const LatticeBaseExpr<V, U>& rhs)
    {
      const V& expr = rhs;
      assert(rhs.num_blocks() == this->num_blocks()
	     && rhs.block_volume() == this->block_volume());
      int start = even ? 0 : this->num_blocks();
      int stop = even ? this->num_blocks() : 2 * this->num_blocks();
      for (int i = start; i < stop; ++i)
	for (int j = 0; j < this->block_volume(); ++j)
	  this->_lattice.datum_ref(i, j) = expr.datum_ref(i, j);
      return *this;
    }

    // Accessor for the data
    U datum_ref(const int i, const int j) const
    { return _lattice.datum_ref(i, j); }
    // Grab the other member variables from one of the arguments.
    const std::vector<int>& lattice_shape() const
    { return _lattice.lattice_shape(); }
    const std::vector<int>& block_shape() const
    { return _lattice.block_shape(); }
    const std::vector<std::vector<int> >& layout() const
    { return _lattice.layout(); }
    const int lattice_volume() const
    { return _lattice.lattice_volume(); }
    const int num_blocks() const
    { return _lattice.num_blocks() / 2; }
    const int block_volume() const
    { return _lattice.block_volume(); }

  private:
    typename SubsetTraits<T, U>::member_type _lattice;
  };



  template <typename T, typename U>
  class LatticeBaseRoll
    : public LatticeBaseExpr<LatticeBaseRoll<T, U>, U>
  {
    // Expression to handle rolling the lattice along a particular axis
  public:
    LatticeBaseRoll(typename SubsetTraits<T, U>::constructor_type lattice,
		    const int dimension, const int shift)
      : _lattice(lattice)
    {
      // Create the nested vector of rolled indices

      // Check that the dimension is sane
      assert(dimension >= 0 && dimension < this->lattice_shape().size());
      assert(abs(shift) < this->lattice_shape()[dimension]);

      // Allocate space
      this->_rolled_layout.resize(lattice.num_blocks());
      for (auto& elem : this->_rolled_layout) {
	elem.resize(this->block_volume());
	for (auto& subelem : elem)
	  subelem.resize(2);
      }

      // Compute the offset for the given dimension and shift
      // This is the shift required for the lexicographic index
      // in the given dimension
      int axis_shift = std::accumulate(this->lattice_shape().begin()
				       + dimension + 1,
				       this->lattice_shape().end(), 1,
				       std::multiplies<int>());
      // This is the correction we'll need if we go outside the lattice
      // volume ( < 0 or > vol)
      int axis_modulo = axis_shift * this->lattice_shape()[dimension];
      // Then we multiply by the actual shift.
      axis_shift *= -shift;

      for (int i = 0; i < this->lattice_volume(); ++i) {
	// Do the shift
	int rolled_index = i + axis_shift;
	if (i % axis_modulo + axis_shift >= axis_modulo)
	  rolled_index -= axis_modulo;
	else if (i % axis_modulo + axis_shift < 0)
	  rolled_index += axis_modulo;
	// Now assign the new coordinates to the _rolled_layout member variable
	std::vector<int> old_coords = this->layout()[i];
	this->_rolled_layout[old_coords[0]][old_coords[1]]
	  = this->layout()[rolled_index];
      }
    }

    LatticeBaseRoll& operator=(const LatticeBaseRoll<T, U>& rhs)
    {
      typedef LatticeBaseExpr<LatticeBaseRoll<T, U>, U> cast_type;
      return operator=(static_cast<const cast_type&>(rhs));
    }

    template <typename V>
    LatticeBaseRoll& operator=(const LatticeBaseExpr<V, U>& rhs)
    {
      const V& expr = rhs;
      assert(rhs.num_blocks() == this->num_blocks()
	     && rhs.block_volume() == this->block_volume());
      for (int i = 0; i < this->num_blocks(); ++i)
	for (int j = 0; j < this->block_volume(); ++j)
	  this->_lattice.datum_ref(this->_rolled_layout[i][j][0],
				   this->_rolled_layout[i][j][1])
	    = expr.datum_ref(i, j);
      return *this;
    }

    // Accessor for the data
    U datum_ref(const int i, const int j) const
    {
      return _lattice.datum_ref(this->_rolled_layout[i][j][0],
				this->_rolled_layout[i][j][1]);
    }
    // Grab the other member variables from one of the arguments.
    const std::vector<int>& lattice_shape() const
    { return _lattice.lattice_shape(); }
    const std::vector<int>& block_shape() const
    { return _lattice.block_shape(); }
    const std::vector<std::vector<int> >& layout() const
    { return _lattice.layout(); }
    const int lattice_volume() const
    { return _lattice.lattice_volume(); }
    const int num_blocks() const
    { return _lattice.num_blocks(); }
    const int block_volume() const
    { return _lattice.block_volume(); }

  private:
    typename SubsetTraits<T, U>::member_type _lattice;
    std::vector<std::vector<std::vector<int> > > _rolled_layout;
  };



  template <typename T, typename U>
  class LatticeBaseConst
    : public LatticeBaseExpr<LatticeBaseConst<T, U>, U>
  {
    // Create a constant expression for use in scalar multiplication and
    // division.
  public:
    LatticeBaseConst(const T& scalar)
      : _scalar(scalar)
    { }

    // Accessor for the data
    const T& datum_ref(const int i, const int j) const
    { return _scalar; }
  private:
    T const& _scalar;
  };



  template <typename T, typename U, typename V, typename O>
  class LatticeBaseBinary
    : public LatticeBaseExpr<LatticeBaseBinary<T, U, V, O>, V>
  {
    // Expression sub-class: subtraction.
  public:
    // Constructed from two other expressions: the bits either side of the minus
    // sign.
    LatticeBaseBinary(const LatticeBaseExpr<T, V>& lhs,
		      const LatticeBaseExpr<U, V>& rhs)
      : _lhs(lhs), _rhs(rhs)
    { }
    // Here we denote the actual arithmetic operation.
    V datum_ref(const int i, const int j) const
    { return O::apply(_lhs.datum_ref(i, j), _rhs.datum_ref(i, j)); }
    // Grab the other member variables from one of the arguments.
    const std::vector<int>& lattice_shape() const
    { return _lhs.lattice_shape(); }
    const std::vector<int>& block_shape() const
    { return _lhs.block_shape(); }
    const std::vector<std::vector<int> >& layout() const
    { return _lhs.layout(); }
    const int lattice_volume() const
    { return _lhs.lattice_volume(); }
    const int num_blocks() const
    { return _lhs.num_blocks(); }
    const int block_volume() const
    { return _lhs.block_volume(); }
  private:
    // The members - what we're subtracting (rhs) from something else (lhs)
    typename BinaryTraits<T>::type _lhs;
    typename BinaryTraits<U>::type _rhs;
  };



  struct Plus
  {
    template <typename T, typename U>
    static T apply(const T& lhs, const U& rhs)
    { return lhs + rhs; }
  };



  struct Minus
  {
    template <typename T, typename U>
    static T apply(const T& lhs, const U& rhs)
    { return lhs - rhs; }
  };



  struct Multiplies
  {
    template <typename T, typename U>
    static T apply(const T& lhs, const U& rhs)
    { return lhs * rhs; }
  };



  struct Divides
  {
    template <typename T, typename U>
    static T apply(const T& lhs, const U& rhs)
    { return lhs / rhs; }
  };



  // Now we add some syntactic sugar by overloading the various arithmetic
  // operators corresponding to the expressions we've implemented above.
  template <typename T, typename U, typename V>
  const LatticeBaseBinary<T, U, V, Minus>
  operator-(const LatticeBaseExpr<T, V>& lhs,
	    const LatticeBaseExpr<U, V>& rhs)
  {
    return LatticeBaseBinary<T, U, V, Minus>(lhs, rhs);
  }



  template <typename T, typename U, typename V>
  const LatticeBaseBinary<T, U, V, Plus>
  operator+(const LatticeBaseExpr<T, V>& lhs,
	    const LatticeBaseExpr<U, V>& rhs)
  {
    return LatticeBaseBinary<T, U, V, Plus>(lhs, rhs);
  }



  template <typename T, typename U, typename V>
  const LatticeBaseBinary<T, LatticeBaseConst<V, U>, V, Multiplies>
  operator*(const V& scalar, const LatticeBaseExpr<T, U>& lattice)
  {
    return LatticeBaseBinary<T, LatticeBaseConst<V, U>, V, Multiplies>
      (lattice, LatticeBaseConst<V, U>(scalar));
  }



  template <typename T, typename U, typename V>
  const LatticeBaseBinary<T, LatticeBaseConst<V, U>, V, Multiplies>
  operator*(const LatticeBaseExpr<T, U>& lattice, const V& scalar)
  {
    return LatticeBaseBinary<T, LatticeBaseConst<V, U>, V, Multiplies>
      (lattice, LatticeBaseConst<V, U>(scalar));
  }



  template <typename T, typename U, typename V>
  const LatticeBaseBinary<T, LatticeBaseConst<V, U>, V, Divides>
  operator/(const LatticeBaseExpr<T, U>& lattice, const V& scalar)
  {
    return LatticeBaseBinary<T, LatticeBaseConst<V, U>, V, Divides>
      (lattice, LatticeBaseConst<V, U>(scalar));
  }
}

#endif
