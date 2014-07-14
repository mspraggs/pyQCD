#ifndef LATTICE_ARRAY_EXP_HPP
#define LATTICE_ARRAY_EXP_HPP

/* This file provides expression templates for the LatticeArray class, so that
 * temporaries do not need to be created when performing arithmetic operations.
 *
 * The idea is to use template metaprogramming to represent arithmetic
 * operations at compile time so that the expression can be compiled down to
 * expressions that iterate over the elements of the LatticeArray, thus negating
 * the need for allocating temporaries.
 */


namespace pyQCD
{
  template <typename T, typename U>
  class LatticeArrayExpr
  {
    // This is the main expression class from which all others are derived. It
    // It takes two template arguments: the type that's involved in expressions
    // and the site type used in LatticeArray. This allows expressions to be
    // abstracted to a nested hierarchy of types. When the compiler goes through
    // and does it's thing, the definitions of the operations within these
    // template classes are all spliced together.
  public:
    // Here we use curiously recurring template patterns to call functions in
    // the LatticeArray type.
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

    operator T&()
    { return static_cast<T&>(*this); }
    operator T const&() const
    { return static_cast<const T&>(*this); }
  };

  template <typename T, typename U, typename V>
  class LatticeArrayDiff
    : public LatticeArrayExpr<LatticeArrayDiff<T, U, V>, V>
  {
    // First expression sub-class: subtraction.
  public:
    // Constructed from two other expressions: the bits either side of the minus
    // sign.
    LatticeArrayDiff(LatticeArrayExpr<T, V> const& lhs,
		     LatticeArrayExpr<U, V> const& rhs)
      : _lhs(lhs), _rhs(rhs)
    { }
    // Here we denote the actual arithmetic operation.
    V datum_ref(const int i, const int j) const
    { return _lhs.datum_ref(i, j) - _rhs.datum_ref(i, j); }
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
    T const& _lhs;
    U const& _rhs;
  };

  template <typename T, typename U, typename V>
  class LatticeArraySum
    : public LatticeArrayExpr<LatticeArraySum<T, U, V>, V>
  {
    // Next: addition
  public:
    LatticeArraySum(const LatticeArrayExpr<T, V>& lhs,
		    const LatticeArrayExpr<U, V>& rhs)
      : _lhs(lhs), _rhs(rhs)
    { }
    // Again, denote the operation here.
    V datum_ref(const int i, const int j) const
    { return _lhs.datum_ref(i, j) + _rhs.datum_ref(i, j); }
    // Get our members from the sub-expressions.
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
    // And the expressions we're summing.
    const T& _lhs;
    const U& _rhs;
  };

  template <typename T, typename U, typename V>
  class LatticeArrayMult
    : public LatticeArrayExpr<LatticeArrayMult<T, U, V>, U>
  {
    // Scalar multiplication. The scalar type is templated.
  public:
    LatticeArrayMult(const V& scalar, const LatticeArrayExpr<T, U>& lattice)
      : _scalar(scalar), _lattice(lattice)
    { }
    // Define the operation
    U datum_ref(const int i, const int j) const
    { return _scalar * _lattice.datum_ref(i, j); }
    // Once again... get member variables.
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
    const V& _scalar;
    const T& _lattice;
  };

  template <typename T, typename U, typename V>
  class LatticeArrayDiv
    : public LatticeArrayExpr<LatticeArrayDiv<T, U, V>, U>
  {
    // Scalar division.
  public:
    LatticeArrayDiv(const V& scalar, const LatticeArrayExpr<T, U>& lattice)
      : _scalar(scalar), _lattice(lattice)
    { }
    U datum_ref(const int i, const int j) const
    { return _lattice.datum_ref(i, j) / _scalar; }

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
    const V& _scalar;
    const T& _lattice;
  };



  // Now we add some syntactic sugar by overloading the various arithmetic
  // operators corresponding to the expressions we've implemented above.
  template <typename T, typename U, typename V>
  const LatticeArrayDiff<T, U, V>
  operator-(const LatticeArrayExpr<T, V>& lhs,
	    const LatticeArrayExpr<U, V>& rhs)
  {
    return LatticeArrayDiff<T, U, V>(lhs, rhs);
  }



  template <typename T, typename U, typename V>
  const LatticeArrayDiff<T, U, V>
  operator+(const LatticeArrayExpr<T, V>& lhs,
	    const LatticeArrayExpr<U, V>& rhs)
  {
    return LatticeArrayDiff<T, U, V>(lhs, rhs);
  }



  template <typename T, typename U, typename V>
  const LatticeArrayMult<T, U, V>
  operator*(const V& scalar, const LatticeArrayExpr<T, U>& lattice)
  {
    return LatticeArrayMult<T, U, V>(scalar, lattice);
  }



  template <typename T, typename U, typename V>
  const LatticeArrayMult<T, U, V>
  operator*(const LatticeArrayExpr<T, U>& lattice, const V& scalar)
  {
    return LatticeArrayMult<T, U, V>(scalar, lattice);
  }



  template <typename T, typename U, typename V>
  const LatticeArrayDiv<T, U, V>
  operator/(const LatticeArrayExpr<T, U>& lattice, const V& scalar)
  {
    return LatticeArrayDiv<T, U, V>(scalar, lattice);
  }
}

#endif
