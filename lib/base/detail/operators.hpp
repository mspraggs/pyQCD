#ifndef OPERATORS_HPP
#define OPERATORS_HPP

/* Defines operators for use in Array expression classes */


struct Plus
{
  template <typename T1, typename T2>
  static auto apply(const T1& lhs, const T2& rhs) -> decltype(lhs + rhs)
  { return lhs + rhs; }
};


struct Minus
{
  template <typename T1, typename T2>
  static auto apply(const T1& lhs, const T2& rhs) -> decltype(lhs - rhs)
  { return lhs - rhs; }
};


struct Multiplies
{
  template <typename T1, typename T2>
  static auto apply(const T1& lhs, const T2& rhs) -> decltype(lhs * rhs)
  { return lhs * rhs; }
};


struct Divides
{
  template <typename T1, typename T2>
  static auto apply(const T1& lhs, const T2& rhs) -> decltype(lhs / rhs)
  { return lhs / rhs; }
};

#endif