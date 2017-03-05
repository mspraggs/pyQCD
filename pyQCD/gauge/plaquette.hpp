#ifndef PYQCD_PLAQUETTE_HPP
#define PYQCD_PLAQUETTE_HPP
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
 * Created by Matt Spraggs on 09/09/16.
 *
 *
 * Below we define functions to compute a specific plaquette and the average
 * plaquette for a given gauge field.
 */

#include <core/qcd_types.hpp>


namespace pyQCD
{
  namespace gauge
  {
    template <typename Real, int Nc>
    Real plaquette(const LatticeColourMatrix<Real, Nc>& gauge_field,
                   const Int site, const Int mu, const Int nu)
    {
      Site site_coords_orig = gauge_field.layout().compute_site_coords(site);
      Site site_coords_mu = site_coords_orig;
      Site site_coords_nu = site_coords_orig;
      site_coords_mu[mu] += 1;
      site_coords_nu[nu] += 1;
      gauge_field.layout().sanitize_site_coords(site_coords_mu);
      gauge_field.layout().sanitize_site_coords(site_coords_nu);

      auto mat = gauge_field(site_coords_orig, mu);
      mat *= gauge_field(site_coords_mu, nu);
      mat *= gauge_field(site_coords_nu, mu).adjoint();
      mat *= gauge_field(site_coords_orig, nu).adjoint();

      return mat.trace().real() / Nc;
    }


    template <typename Real, int Nc>
    Real average_plaquette(const LatticeColourMatrix<Real, Nc>& gauge_field)
    {
      Real total = 0.0;
      Int num = gauge_field.num_dims() * (gauge_field.num_dims() - 1) / 2
        * gauge_field.volume();
      for (Int site = 0; site < gauge_field.volume(); ++site) {
        for (Int mu = 0; mu < gauge_field.num_dims(); ++mu) {
          for (Int nu = mu + 1; nu < gauge_field.num_dims(); ++nu) {
            total += plaquette(gauge_field, site, mu, nu);
          }
        }
      }
      return total / num;
    }
  }
}

#endif //PYQCD_PLAQUETTE_HPP
