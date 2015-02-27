#ifndef LATTICE_HPP
#define LATTICE_HPP

/* This file declares and defines the Lattice class. This is basically an Array
 * but with a Layout member specifying the relationship between the sites and
 * the Array index.
 */

#include <vector>

#include "array.hpp"
#include "layout.hpp"


namespace pyQCD
{
  template <typename T, template <typename> class Alloc = std::allocator>
  class Lattice : public Array<T, Alloc>
  {
  public:
    Lattice(const Layout* shape, const T& val)
      : layout_(shape), Array(layout_->volume(), val)
    {}
    Lattice(const Lattice<T, Alloc>& lattice);
    Lattice(Lattice<T, Alloc>&& lattice) = default;

    T& operator()(const int i) { }
    const T& operator()(const int i) const { }

    Lattice<T, Alloc>& operator=(const Lattice<T, Alloc>& lattice);
    Lattice<T, Alloc>& operator=(Lattice<T, Alloc>&& lattice);

  protected:
    Layout* layout_;
  };
}

#endif