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
 * Created by Matt Spraggs on 13/01/17.
 *
 * End-to-end test of gauge field update using Wilson action and heatbath
 * algorithm.
 */

#include <algorithms/heatbath.hpp>
#include <gauge/plaquette.hpp>
#include <gauge/rectangle.hpp>
#include <gauge/wilson_action.hpp>

#include "helpers.hpp"


TEST_CASE("End-to-end heatbath test with Wilson action")
{
  using GaugeField = pyQCD::gauge::Action<double, 3>::GaugeField;
  using GaugeLink = pyQCD::gauge::Action<double, 3>::GaugeLink;

  const pyQCD::LexicoLayout layout({4, 4, 4, 4});
  GaugeField gauge_field(layout, GaugeLink::Identity(), 4);

  pyQCD::gauge::WilsonAction<double, 3> action(5.5, layout);

  pyQCD::Heatbath<double, 3> updater(layout, action);

  std::vector<std::size_t> seeds(layout.volume(), 0);
  std::iota(seeds.begin(), seeds.end(), 0);
  pyQCD::RandomWrapper::instance(layout).set_seeds(seeds);

  updater.update(gauge_field, 1);

  const double plaquette = pyQCD::gauge::average_plaquette(gauge_field);
  REQUIRE(plaquette == Approx(0.667267594069949).epsilon(1e-13));
  const double rectangle = pyQCD::gauge::average_rectangle(gauge_field);
  REQUIRE(rectangle == Approx(0.4955249692030583).epsilon(1e-13));
}
