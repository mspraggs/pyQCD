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
 * Created by Matt Spraggs on 30/01/17.
 *
 * End-to-end test of gauge field update using Symanzik rectangle-improved
 * action and heatbath algorithm.
 */

#include <algorithms/heatbath.hpp>
#include <gauge/plaquette.hpp>
#include <gauge/rectangle.hpp>
#include <gauge/rectangle_action.hpp>

#include "helpers.hpp"


TEST_CASE("End-to-end heatbath test with rectangle action")
{
  using GaugeField = pyQCD::gauge::Action<double, 3>::GaugeField;
  using GaugeLink = pyQCD::gauge::Action<double, 3>::GaugeLink;

  const pyQCD::LexicoLayout layout({4, 4, 4, 4});
  GaugeField gauge_field(layout, GaugeLink::Identity(), 4);

  const pyQCD::gauge::RectangleAction<double, 3> action(4.41, layout,
                                                        -1.0 / 12.0);

  std::vector<std::size_t> seeds(layout.volume(), 0);
  std::iota(seeds.begin(), seeds.end(), 0);
  pyQCD::RandomWrapper::instance(layout).set_seeds(seeds);

  pyQCD::Heatbath<double, 3> updater(layout, action);

  updater.update(gauge_field, 1);

  const double plaquette = pyQCD::gauge::average_plaquette(gauge_field);
  REQUIRE(plaquette == Approx(0.7160328906006762).epsilon(1e-13));
  const double rectangle = pyQCD::gauge::average_rectangle(gauge_field);
  REQUIRE(rectangle == Approx(0.547103350169803).epsilon(1e-13));
}