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

namespace pyQCD {
  class Communicator {
    Communicator();

  public:

    static Communicator& instance();

    void init(MPI_Comm& comm);

    template<typename T>
    static void send(const T& data);
    template<typename T>
    static void recv(const T& data);

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
}

#endif