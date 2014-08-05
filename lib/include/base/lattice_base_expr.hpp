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




namespace pyQCD
{
  template <typename T, int ndim>
  class LatticeBase;

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

    operator T&()
    { return static_cast<T&>(*this); }
    operator T const&() const
    { return static_cast<const T&>(*this); }
  };



  template <typename T, typename U>
  class LatticeBaseEven
    : public LatticeBaseExpr<LatticeBaseEven<T, U>, U>
  {
    // Generates a reference to the even sites in a lattice
  public:
    // Constructed from a reference to a LatticeBaseExpr.
    LatticeBaseEven(LatticeBaseExpr<T, U>& lattice)
      : _lattice(lattice)
    { }

    LatticeBaseEven& operator=(const LatticeBaseEven<T, U>& rhs)
    {
      return operator=(static_cast<const LatticeBaseExpr<LatticeBaseEven<T, U>,
		       U>& >(rhs));
    }

    template <typename V>
    LatticeBaseEven& operator=(const LatticeBaseExpr<V, U>& rhs)
    {
      const V& expr = rhs;
      assert(rhs.num_blocks() == this->num_blocks()
	     && rhs.block_volume() == this->block_volume());
      for (int i = 0; i < this->num_blocks(); ++i)
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
    T& _lattice;
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
