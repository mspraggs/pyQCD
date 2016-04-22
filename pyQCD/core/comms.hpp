#ifndef PYQCD_COMMS_HPP
#define PYQCD_COMMS_HPP
/*
 * This file is part of pyQCD.
 *
 * pyQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
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
 * Created by Matt Spraggs on 10/02/16.
 *
 *
 * Clean C++ interface to MPI routines
 */

#include <type_traits>
#include <unordered_map>

#include <mpi.h>

#include "detail/layout_helpers.hpp"

namespace pyQCD {
  class Communicator {
    Communicator();

  public:

    static Communicator& instance();

    void init(MPI_Comm& comm);

    int size() const { return size_; }
    int rank() const { return rank_; }
    const MPI_Comm& comm() const { return *comm_; }

  private:
    MPI_Comm* comm_;
    int size_, rank_;
    bool initialized_;
    template <typename T>
    void ensure_mpi_type();
    std::unordered_map<const std::type_info*, MPI_Datatype> mpi_types_;
  };

  template <typename T>
  void Communicator::ensure_mpi_type()
  {
    // Ensure that we have an MPI_Datatype instance in mpi_types
  }

  template <typename T>
  struct MpiType
  {
    static MPI_Datatype make(const detail::Int vec_size = 1);
  };

  template <>
  struct MpiType<float>
  {
    static MPI_Datatype make(const detail::Int vec_size = 1)
    {
      MPI_Datatype ret;
      MPI_Type_contiguous(vec_size, MPI_FLOAT, &ret);
      return ret;
    }
  };

  template <>
  struct MpiType<double>
  {
    static MPI_Datatype make(const detail::Int vec_size = 1)
    {
      MPI_Datatype ret;
      MPI_Type_contiguous(vec_size, MPI_DOUBLE, &ret);
      return ret;
    }
  };

  template <typename T>
  struct MpiType<std::complex<T>>
  {
    static MPI_Datatype make(const detail::Int vec_size = 1)
    {
      auto inner_type = MpiType<T>::make();
      MPI_Datatype ret;
      MPI_Type_contiguous(vec_size * 2, inner_type, &ret);
      return ret;
    }
  };

  template <typename T, int N, int M>
  struct MpiType<Eigen::Matrix<T, N, M>>
  {
    static MPI_Datatype make(const detail::Int vec_size = 1)
    {
      auto inner_type = MpiType<T>::make();
      MPI_Datatype ret;
      MPI_Type_contiguous(vec_size * N * M, inner_type, &ret);
      return ret;
    }
  };
}

#endif