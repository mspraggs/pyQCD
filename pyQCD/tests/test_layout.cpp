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
  typedef pyQCD::LexicoLayout Layout;

#ifdef USE_MPI

  MPI_Comm comm;
  int p[] = {2, 2, 1, 1};
  int periodic[] = {0, 0, 0, 0};

  int err = MPI_Cart_create(MPI_COMM_WORLD, 4, p, periodic, 0, &comm);

  pyQCD::Site partition(p, p + 4);

  pyQCD::Communicator::instance().init(comm);

  Layout layout({8, 4, 4, 4}, partition);
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