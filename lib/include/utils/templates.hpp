#ifndef TEMPLATES_HPP
#define TEMPLATES_HPP

/* This file provides various utilities for use in template metaprogramming,
 * including extending the type_traits utilities where necessary.
 */

/* The following structs allow us to determine whether a given class is derived
 * from a template base class. I've adapted this from the example by bluescarni
 * from http://stackoverflow.com/questions/5997956		\
 * /how-to-determine-if-a-type-is-derived-from-a-template-class
 */

#include <type_traits>

namespace pyQCD
{
  template <template <typename...> class F>
  struct conversion_tester_type_temp
  {
    template <typename... Args>
    conversion_tester_type_temp (const F<Args...>&);
  };
  
  template <class From, template <typename...> class To>
  struct is_instance_of_type_temp
  {
    static const bool value
    = std::is_convertible<From, conversion_tester_type_temp<To> >::value;
  };



  template <template <int, int...> class F>
  struct conversion_tester_int_temp
  {
    template <int N, int... Args>
    conversion_tester_int_temp (const F<N, Args...>&);
  };

  template <class From, template <int, int...> class To>
  struct is_instance_of_int_temp
  {
    static const bool value
    = std::is_convertible<From, conversion_tester_int_temp<To> >::value;
  };
  
  
  
  template <template <typename, template <typename> class> class F>
  struct conversion_tester_Array
  {
    template <typename T, template <typename> class A>
    conversion_tester_Array(const F<T, A>&);
  };
  
  template <class From,
    template <typename, template <typename> class> class To>
  struct is_instance_of_Array
  {
    static const bool value
    = std::is_convertible<From, conversion_tester_Array<To> >::value;
  };
}

#endif
