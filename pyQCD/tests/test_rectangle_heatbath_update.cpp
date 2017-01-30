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

#define CATCH_CONFIG_MAIN

#include <algorithms/heatbath.hpp>
#include <gauge/plaquette.hpp>
#include <gauge/rectangle.hpp>
#include <gauge/rectangle_action.hpp>

#include "helpers.hpp"


TEST_CASE("End-to-end heatbath test with Wilson action")
{
  typedef pyQCD::gauge::Action<double, 3>::GaugeField GaugeField;
  typedef pyQCD::gauge::Action<double, 3>::GaugeLink GaugeLink;

  pyQCD::LexicoLayout layout({8, 8, 8, 8});
  GaugeField gauge_field(layout, GaugeLink::Identity(), 4);

  pyQCD::gauge::RectangleAction<double, 3> action(4.41, layout, -1.0 / 12.0);

  double avg_plaquette = 1.1;
  for (unsigned int i = 0; i < 5; ++i) {
    heatbath_update(gauge_field, action, 1);
    double new_avg_plaquette = pyQCD::gauge::average_plaquette(gauge_field);
    // Statistically it's highly probable there will be a monotonic decrease in
    // the average plaquette over the first five updates.
    REQUIRE (new_avg_plaquette < avg_plaquette);
    avg_plaquette = new_avg_plaquette;
  }

  for (unsigned int i = 0; i < 9; ++i) {
    heatbath_update(gauge_field, action, 10);
  }

  avg_plaquette = pyQCD::gauge::average_plaquette(gauge_field);

  REQUIRE(avg_plaquette < 0.63);
  REQUIRE(avg_plaquette > 0.61);
}