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
 * Tests for the Layout class.
 */

#define CATCH_CONFIG_RUNNER

#include <core/layout.hpp>

#include "helpers.hpp"


TEST_CASE("MPI offset test") {
  std::vector<bool> need_comms{true, false, true, false};
  auto mpi_offsets = pyQCD::detail::generate_mpi_offsets(1, need_comms);
  REQUIRE(mpi_offsets.size() == 4);

  need_comms = {true, true, false};
  mpi_offsets = pyQCD::detail::generate_mpi_offsets(2, need_comms);
  REQUIRE(mpi_offsets.size() == 8);

  need_comms = {true, true, false, true, true};
  mpi_offsets = pyQCD::detail::generate_mpi_offsets(2, need_comms);
  REQUIRE(mpi_offsets.size() == 32);
}


TEST_CASE("LexicoLayout test") {
  typedef pyQCD::Layout Layout;

#ifdef USE_MPI

  MPI_Comm comm;
  int p[] = {2, 2, 1, 1};
  int periodic[] = {1, 1, 1, 1};

  MPI_Cart_create(MPI_COMM_WORLD, 4, p, periodic, 1, &comm);

  pyQCD::Site partition(p, p + 4);
  pyQCD::Communicator::instance().init(comm);

  Layout layout({8, 4, 4, 4}, partition, 1, 2);

  REQUIRE (layout.local_size() == 384);
  REQUIRE (layout.local_volume() == 128);
  REQUIRE (layout.global_volume() == 512);
  REQUIRE ((layout.local_shape() == pyQCD::Site{4, 2, 4, 4}));
  REQUIRE ((layout.global_shape() == pyQCD::Site{8, 4, 4, 4}));

  REQUIRE (layout.buffer_volume(0) == 32);
  REQUIRE (layout.buffer_volume(1) == 64);
  REQUIRE (layout.buffer_volume(4) == 16);

  for (pyQCD::Int axis = 0; axis < 2 * layout.num_dims(); ++axis) {
    REQUIRE (layout.buffer_indices(axis, 1).size() == ((axis < 4) ? 1 : 0));
    REQUIRE (layout.buffer_indices(axis, 2).size() == ((axis < 4) ? 2 : 0));
  }
  // Test that the buffer_map_ has been constructed correctly, starting with
  // one hop
  REQUIRE (layout.buffer_indices(0, 1)[0] == 0);
  REQUIRE (layout.buffer_indices(2, 1)[0] == 1);
  REQUIRE (layout.buffer_indices(3, 1)[0] == 2);
  REQUIRE (layout.buffer_indices(1, 1)[0] == 3);
  // Now two mpi hops
  REQUIRE (layout.buffer_indices(0, 2)[0] == 4);
  REQUIRE (layout.buffer_indices(0, 2)[1] == 5);
  REQUIRE (layout.buffer_indices(2, 2)[0] == 4);
  REQUIRE (layout.buffer_indices(2, 2)[1] == 6);
  REQUIRE (layout.buffer_indices(1, 2)[0] == 6);
  REQUIRE (layout.buffer_indices(1, 2)[1] == 7);
  REQUIRE (layout.buffer_indices(3, 2)[0] == 5);
  REQUIRE (layout.buffer_indices(3, 2)[1] == 7);

  // Now check that the mapping between lexicographic index and array index
  // is computed correctly
  // These test the unbuffered site indices (i.e. those that don't exist within
  // a halo).
  REQUIRE (layout.get_array_index(80) == 0);
  REQUIRE (layout.get_array_index(95) == 15);
  REQUIRE (layout.get_array_index(239) == 95);
  // These just test that the halo buffers are indexed correctly
  REQUIRE (layout.get_array_index(0) == 320);
  REQUIRE (layout.get_array_index(14) == 334);
  REQUIRE (layout.get_array_index(383) == 383);

  REQUIRE (layout.buffered_array_indices(0).size() == 32);
  REQUIRE (layout.buffered_array_indices(1).size() == 64);
  REQUIRE (layout.buffered_array_indices(4).size() == 16);

  REQUIRE (layout.buffered_array_indices(0)[0] == 272);
  REQUIRE (layout.buffered_array_indices(0)[10] == 282);
  REQUIRE (layout.buffered_array_indices(1)[0] == 96);
  REQUIRE (layout.buffered_array_indices(1)[10] == 106);
  REQUIRE (layout.buffered_array_indices(4)[0] == 288);
  REQUIRE (layout.buffered_array_indices(4)[10] == 298);
#else
  Layout layout({8, 4, 4, 4});
#endif
/*
  for (int i = 0; i < 512; ++i) {
    REQUIRE (layout.get_array_index(i) == i);
    REQUIRE (layout.get_site_index(i) == i);
  }
  REQUIRE (layout.get_array_index(pyQCD::Site{4, 3, 2, 1})
             == 313);
  REQUIRE (layout.volume() == 512);
  REQUIRE (layout.num_dims() == 4);
  REQUIRE ((layout.shape() == pyQCD::Site{8, 4, 4, 4}));
*/
}


int main(int argc, char * argv[]) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}