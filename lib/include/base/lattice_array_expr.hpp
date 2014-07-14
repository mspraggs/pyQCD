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
  public:
    U& datum_ref(const int i, const int j)
    { return static_cast<T&>(*this).datum_ref(i, j); }
    const U& datum_ref(const int i, const int j) const
    { return static_cast<const T&>(*this).datum_ref(i, j); }

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
  public:
    LatticeArrayDiff(LatticeArrayExpr<T, V> const& lhs,
		     LatticeArrayExpr<U, V> const& rhs)
      : _lhs(lhs), _rhs(rhs)
    { }
    
    V datum_ref(const int i, const int j) const
    { return _lhs.datum_ref(i, j) - _rhs.datum_ref(i, j); }

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
    T const& _lhs;
    U const& _rhs;
  };

  template <typename T, typename U, typename V>
  class LatticeArraySum
    : public LatticeArrayExpr<LatticeArraySum<T, U, V>, V>
  {
  public:
    LatticeArraySum(const LatticeArrayExpr<T, V>& lhs,
		    const LatticeArrayExpr<U, V>& rhs)
      : _lhs(lhs), _rhs(rhs)
    { }
    
    V datum_ref(const int i, const int j) const
    { return _lhs.datum_ref(i, j) + _rhs.datum_ref(i, j); }

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
    const T& _lhs;
    const U& _rhs;
  };

  template <typename T, typename U, typename V>
  class LatticeArrayMult
    : public LatticeArrayExpr<LatticeArrayMult<T, U, V>, U>
  {
  public:
    LatticeArrayMult(const V& scalar, const LatticeArrayExpr<T, U>& lattice)
      : _scalar(scalar), _lattice(lattice)
    { }
    U datum_ref(const int i, const int j) const
    { return _scalar * _lattice.datum_ref(i, j); }

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
