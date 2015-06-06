#ifndef LAYOUT_HPP
#define LAYOUT_HPP

/* This file provides a base Layout classes and derived classes that specify
 * the layout of lattice sites. These classes are then used in Lattice objects
 * and their derived types to specify the relationship between e
 */

#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>


namespace pyQCD
{
  class Layout
  {
  public:
    Layout() { }
    template <typename Fn>
    Layout(const std::vector<unsigned int>& lattice_shape,
      Fn compute_array_index)
      : num_dims_(static_cast<unsigned int>(lattice_shape.size())),
        lattice_shape_(lattice_shape)
    {
      // Constructor create arrays of site/array indices
      lattice_volume_ = std::accumulate(lattice_shape.begin(),
                                        lattice_shape.end(),
                                        unsigned(1),
                                        std::multiplies<unsigned int>());

      array_indices_.resize(lattice_volume_);
      site_indices_.resize(lattice_volume_);
      for (unsigned int site_index = 0;
           site_index < lattice_volume_;
           ++site_index) {
        unsigned int array_index = compute_array_index(site_index);
        array_indices_[site_index] = array_index;
        site_indices_[array_index] = site_index;
      }
    }
    virtual ~Layout() = default;

    template <typename T,
      typename std::enable_if<not std::is_integral<T>::value>::type* = nullptr>
    inline unsigned int get_array_index(const T& site) const;
    inline unsigned int get_array_index(const unsigned int site_index) const
    { return array_indices_[site_index]; }
    inline unsigned int get_site_index(const unsigned int array_index) const
    { return site_indices_[array_index]; }

    unsigned int volume() const { return lattice_volume_; }
    unsigned int num_dims() const { return num_dims_; }
    const std::vector<unsigned int>& lattice_shape() const
    { return lattice_shape_; }

  private:
    unsigned int num_dims_, lattice_volume_;
    std::vector<unsigned int> lattice_shape_;
    // array_indices_[site_index] -> array_index
    std::vector<unsigned int> array_indices_;
    // site_indices_[array_index] -> site_index
    std::vector<unsigned int> site_indices_;
  };


  class LexicoLayout : public Layout
  {
  public:
    LexicoLayout() { }
    LexicoLayout(const std::vector<unsigned int>& shape)
      : Layout(shape, [] (const unsigned int i) { return i; })
    { }
  };


  template <typename T,
    typename std::enable_if<not std::is_integral<T>::value>::type*>
  inline unsigned int Layout::get_array_index(const T& site) const
  {
    // Compute the lexicographic index of the specified site and use it to
    // to get the array index (coordinate at site[0] varies slowest, that at
    // site[ndim - 1] varies fastest
    int site_index = 0;
    for (unsigned int i = 0; i < num_dims_; ++i) {
      site_index *= lattice_shape_[i];
      site_index += site[i];
    }
    return array_indices_[site_index];
  }
}

#endif