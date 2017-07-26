#ifndef PYQCD_ALIGNED_ALLOCATOR_HPP
#define PYQCD_ALIGNED_ALLOCATOR_HPP
/*
 * This file is part of pyQCD.
 *
 * pyQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pyQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 *
 * Created by Matt Spraggs on 27/01/17.
 *
 * Defined allocator for use with std::vector to enforce 16 byte alignment.
 */

#include <memory>


namespace pyQCD
{
  namespace detail
  {
    template <typename T>
    class aligned_allocator : public std::allocator<T>
    {
    public:
      using size_type       = typename std::allocator<T>::size_type;
      using difference_type = typename std::allocator<T>::difference_type;
      using pointer         = typename std::allocator<T>::pointer;
      using const_pointer   = typename std::allocator<T>::const_pointer;
      using reference       = typename std::allocator<T>::reference;
      using const_reference = typename std::allocator<T>::const_reference;

      template <typename U>
      struct rebind { typedef aligned_allocator<U> other; };

      using std::allocator<T>::allocator;

      pointer allocate(size_type num, const void* = nullptr)
      {
        return new T[num + alignof(T)];
      }

      void deallocate(pointer ptr, size_type)
      {
        delete[] ptr;
      }
    };
  }
}



#endif //PYQCD_ALIGNED_ALLOCATOR_HPP
