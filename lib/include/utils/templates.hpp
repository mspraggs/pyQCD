#ifndef TEMPLATES_HPP
#define TEMPLATES_HPP

/* This file provides various utilities for use in template metaprogramming,
 * including extending the type_traits utilities where necessary.
 */

// The following structs allow us to determine whether a given class is derived
// from a template base class. I've adapted this from the example by bluescarni
// from http://stackoverflow.com/questions/5997956 \
// /how-to-determine-if-a-type-is-derived-from-a-template-class

template <template <typename...> class F>
struct conversion_tester
{
  template <typename... Args>
  conversion_tester (const F<Args...>&);
};

template <class From, template <typename...> class To>
struct is_instance_of
{
  static const bool value
  = std::is_convertible<From, conversion_tester<To> >::value;
};



template <template <int...> class F>
struct conversion_tester
{
  template <int... Args>
  conversion_tester (const F<Args...>&);
};

template <class From, template <int...> class To>
struct is_instance_of
{
  static const bool value
  = std::is_convertible<From, conversion_tester<To> >::value;
};



// Overloaded for cases like LatticeBase<T, ndim>
template <template <typename, int> class F>
struct conversion_tester
{
  template <typename T, int N>
  conversion_tester (const F<T, N>&);
};

template <class From, template <typename, int> class To>
struct is_instance_of
{
  static const bool value
  = std::is_convertible<From, conversion_tester<To> >::value;
};

#endif
