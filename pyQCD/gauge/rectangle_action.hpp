#ifndef PYQCD_GAUGE_RECTANGLE_ACTION_HPP
#define PYQCD_GAUGE_RECTANGLE_ACTION_HPP

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
 * Created by Matt Spraggs on 28/01/17.
 *
 * Generic rectangle-improved gauge action implementation.
 */

#include "gauge_action.hpp"


namespace pyQCD
{
  namespace gauge
  {
    template<typename Real, int Nc>
    class RectangleAction : public Action<Real, Nc>
    {
    public:
      RectangleAction(const Real beta, const Layout& layout, const Real c1);

      typename Action<Real, Nc>::GaugeLink compute_staples(
          const typename Action<Real, Nc>::GaugeField& gauge_field,
          const Int link_index) const;

      Real local_action(
          const typename Action<Real, Nc>::GaugeField& gauge_field,
          const Int link_index) const;

    private:
      std::vector<std::vector<Int>> links_;
      Real c0_, c1_;
    };


    template <typename Real, int Nc>
    RectangleAction<Real, Nc>::RectangleAction(const Real beta,
                                               const Layout &layout,
                                               const Real c1)
      : Action<Real, Nc>(beta, layout), c0_(1 - 8.0 * c1), c1_(c1)
    {
      // Determine which link indices belong to which link staples.

      // TODO: Refactor the following to remove code duplication

      auto num_dims = layout.num_dims();
      links_.resize(layout.volume() * num_dims);
      for (Int site_index = 0; site_index < layout.volume(); ++site_index) {
        Site link_coords = layout.compute_site_coords(site_index);

        for (Int d = 0; d < num_dims; ++d) {
          Int link_index = site_index * num_dims + d;
          // Determine which plane the link does not contribute to.
          Site planes(num_dims - 1);
          unsigned int j = 0;
          for (unsigned int i = 0; i < num_dims; ++i) {
            if (d != i) {
              planes[j++] = i;
            }
          }

          // Given two plaquette staples and six rectangle staples per
          // dimension, and three links per plaquette staple and five per
          // rectangle staple, there should be 6 * 5 + 2 * 3 = 36 links
          // per dimension.
          // (N.B. this is more than the number of unique links, as we're double
          // counting those links that partake in the plaquettes)
          links_[link_index].resize((6 + 30) * (num_dims - 1));

          for (unsigned i = 0; i < num_dims - 1; ++i) {
            std::vector<int> link_coords_copy(
                link_coords.begin(), link_coords.end());

            const Int mu = d;
            const Int nu = planes[i];

            /* First compute the loop above the link. Essentially we're
             * interested in:
             *
             * U_\nu (x + \mu) U^\dagger_\mu (x + \nu) U^\dagger_\nu (x)
             */
            // First link is U_\nu (x + \mu)
            link_coords_copy[mu]++;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;
            // Next link is U^\dagger_\mu (x + \nu)
            link_coords_copy[mu]--;
            link_coords_copy[nu]++;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 1]
                = layout.get_array_index(link_coords_copy) * num_dims + mu;
            // Next link is U^\dagger_\nu (x)
            link_coords_copy[nu]--;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 2]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;

            /* Now we want to compute the staple below the link. Essentially:
             *
             * U^\dagger _\nu (x + \mu - \nu) U^\dagger_\mu (x - \nu)
             *   U^\dagger_\nu (x - \nu)
             */
            // First link is U^\dagger_\nu (x + \mu - \nu)
            link_coords_copy[mu]++;
            link_coords_copy[nu]--;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 3]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;
            // Next link is U^\dagger_\mu (x - \nu)
            link_coords_copy[mu]--;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 4]
                = layout.get_array_index(link_coords_copy) * num_dims + mu;
            // Next link is U_\nu (x - \nu)
            links_[link_index][36 * i + 5]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;

            // Now do this all again for the rectangle staples
            link_coords_copy.assign(link_coords.begin(), link_coords.end());

            /* First compute the landscape staple orientated above and ahead
             * of the link, i.e.
             *
             * U_\mu (x + \mu) U_\nu (x + 2 * \mu) U^\dagger_\mu (x + \mu + \nu)
             *   U^\dagger_\mu (x + \nu) U^\dagger_\nu (x)
             */
            // First link is U_\mu (x + \mu)
            link_coords_copy[mu]++;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 6]
                = layout.get_array_index(link_coords_copy) * num_dims + mu;
            // U_\nu (x + 2 * \mu)
            link_coords_copy[mu]++;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 7]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;
            // U^\dagger_\mu (x + \mu + \nu)
            link_coords_copy[nu]++;
            link_coords_copy[mu]--;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 8]
                = layout.get_array_index(link_coords_copy) * num_dims + mu;
            // U^\dagger_\mu (x + \nu)
            links_[link_index][36 * i + 9] = links_[link_index][36 * i + 1];
            // U^\dagger_\nu (x)
            links_[link_index][36 * i + 10] = links_[link_index][36 * i + 2];

            /* Now compute the landscape staple orientated above and behind
             * the link, i.e.
             *
             * U_\nu (x + \mu) U^\dagger_\mu (x + \nu)
             *   U^\dagger_\mu (x - \mu + \nu) U^\dagger_\nu (x - \mu)
             *   U_\mu (x - \mu)
             */
            // First link is U_\nu (x + \mu)
            links_[link_index][36 * i + 11] = links_[link_index][36 * i];
            // U^\dagger_\mu (x + \nu)
            links_[link_index][36 * i + 12] = links_[link_index][36 * i + 1];
            // U^\dagger_\mu (x - \mu + \nu)
            link_coords_copy[mu] -= 2;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 13]
                = layout.get_array_index(link_coords_copy) * num_dims + mu;
            // U^\dagger_\nu (x - \mu)
            link_coords_copy[nu]--;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 14]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;
            // U_\mu (x - \mu)
            links_[link_index][36 * i + 15]
                = layout.get_array_index(link_coords_copy) * num_dims + mu;

            /* Now compute the portrait staple orientated above the link, i.e.
             *
             * U_\nu (x + \mu) U_\nu (x + \mu + \nu) U^\dagger_\mu (x + 2 * \nu)
             *   U^\dagger_\nu (x + \nu) U^\dagger_\nu (x + \nu)
             */
            // First link is U_\nu (x + \mu)
            link_coords_copy[mu] += 2;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 16]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;
            // U_\nu (x + \mu + \nu)
            link_coords_copy[nu]++;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 17]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;
            // U^\dagger_\mu (x + 2 * \nu)
            link_coords_copy[mu]--;
            link_coords_copy[nu]++;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 18]
                = layout.get_array_index(link_coords_copy) * num_dims + mu;
            // U^\dagger_\nu (x + \nu)
            link_coords_copy[nu]--;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 19]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;
            // U^\dagger_\nu (x)
            links_[link_index][36 * i + 20] = links_[link_index][36 * i + 2];

            /* Now compute the landscape staple orientated below and ahead
             * of the link, i.e.
             *
             * U_\mu (x + \mu) U^\dagger_\nu (x + 2 * \mu - \nu)
             *   U^\dagger_\mu (x  + \mu - \nu) U^\dagger_\mu (x - \nu)
             *   U^\dagger_\nu (x - \nu)
             */
            // First link is U_\mu (x + \mu)
            links_[link_index][36 * i + 21] = links_[link_index][36 * i + 6];
            // U^\dagger_\nu (x + 2 * \mu - \nu)
            link_coords_copy[mu] += 2;
            link_coords_copy[nu] -= 2;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 22]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;
            // U^\dagger_\mu (x + \mu - \nu)
            link_coords_copy[mu]--;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 23]
                = layout.get_array_index(link_coords_copy) * num_dims + mu;
            // U^\dagger_\mu (x - \nu)
            links_[link_index][36 * i + 24] = links_[link_index][36 * i + 4];
            // U^\dagger_\nu (x - \nu)
            links_[link_index][36 * i + 25] = links_[link_index][36 * i + 5];

            /* Now compute the landscape staple orientated below and behind
             * the link, i.e.
             *
             * U^\dagger_\nu (x + \mu - \nu) U^\dagger_\mu (x - \nu)
             *   U^\dagger_\mu (x - \mu - \nu) U_\nu (x - \mu - \nu)
             *   U_\mu (x - \mu)
             */
            // First link is U^\dagger_\nu (x + \mu - \nu)
            links_[link_index][36 * i + 26] = links_[link_index][36 * i + 3];
            // U^\dagger_\mu (x - \nu)
            links_[link_index][36 * i + 27] = links_[link_index][36 * i + 4];
            // U^\dagger_\mu (x - \mu - \nu)
            link_coords_copy[mu] -= 2;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 28]
                = layout.get_array_index(link_coords_copy) * num_dims + mu;
            // U^\dagger_\nu (x - \mu - \nu)
            links_[link_index][36 * i + 29]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;
            // U_\mu (x - \mu)
            link_coords_copy[nu]++;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 30]
                = layout.get_array_index(link_coords_copy) * num_dims + mu;

            /* Now compute the portrait staple orientated below the link, i.e.
             *
             * U^\dagger_\nu (x + \mu - \nu) U^\dagger_\nu (x + \mu  - 2 * \nu)
             *   U^\dagger_\mu (x - 2 * \nu) U_\nu (x - 2 * \nu) U_\nu (x - \nu)
             */
            // First link is U^\dagger_\nu (x + \mu - \nu)
            link_coords_copy[mu] += 2;
            link_coords_copy[nu]--;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 31]
                = links_[link_index][36 * i + 3];
            // U^\dagger_\nu (x + \mu - 2 * \nu)
            link_coords_copy[nu]--;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 32]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;
            // U^\dagger_\mu (x - 2 * \nu)
            link_coords_copy[mu]--;
            layout.sanitize_site_coords(link_coords_copy);
            links_[link_index][36 * i + 33]
                = layout.get_array_index(link_coords_copy) * num_dims + mu;
            // U_\nu (x - 2 * \nu)
            links_[link_index][36 * i + 34]
                = layout.get_array_index(link_coords_copy) * num_dims + nu;
            // U_\nu (x - \nu)
            links_[link_index][36 * i + 35] = links_[link_index][36 * i + 5];
          }
        }
      }
    }

    template <typename Real, int Nc>
    typename Action<Real, Nc>::GaugeLink
    RectangleAction<Real, Nc>::compute_staples(
        const typename Action<Real, Nc>::GaugeField& gauge_field,
        const Int link_index) const
    {
      auto ret = Action<Real, Nc>::GaugeLink::Zero().eval();
      auto temp_colour_mat = ret;

      auto num_dims = gauge_field.layout().num_dims();

      for (unsigned i = 0; i < 36 * (num_dims - 1); i += 36) {
        temp_colour_mat = gauge_field[links_[link_index][i]];
        temp_colour_mat *= gauge_field[links_[link_index][i + 1]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 2]].adjoint();

        ret += c0_ * temp_colour_mat;

        temp_colour_mat = gauge_field[links_[link_index][i + 3]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 4]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 5]];

        ret += c0_ * temp_colour_mat;

        temp_colour_mat = gauge_field[links_[link_index][i + 6]];
        temp_colour_mat *= gauge_field[links_[link_index][i + 7]];
        temp_colour_mat *= gauge_field[links_[link_index][i + 8]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 9]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 10]].adjoint();

        ret += c1_ * temp_colour_mat;

        temp_colour_mat = gauge_field[links_[link_index][i + 11]];
        temp_colour_mat *= gauge_field[links_[link_index][i + 12]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 13]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 14]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 15]];

        ret += c1_ * temp_colour_mat;

        temp_colour_mat = gauge_field[links_[link_index][i + 16]];
        temp_colour_mat *= gauge_field[links_[link_index][i + 17]];
        temp_colour_mat *= gauge_field[links_[link_index][i + 18]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 19]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 20]].adjoint();

        ret += c1_ * temp_colour_mat;

        temp_colour_mat = gauge_field[links_[link_index][i + 21]];
        temp_colour_mat *= gauge_field[links_[link_index][i + 22]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 23]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 24]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 25]];

        ret += c1_ * temp_colour_mat;

        temp_colour_mat = gauge_field[links_[link_index][i + 26]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 27]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 28]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 29]];
        temp_colour_mat *= gauge_field[links_[link_index][i + 30]];

        ret += c1_ * temp_colour_mat;

        temp_colour_mat = gauge_field[links_[link_index][i + 31]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 32]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 33]].adjoint();
        temp_colour_mat *= gauge_field[links_[link_index][i + 34]];
        temp_colour_mat *= gauge_field[links_[link_index][i + 35]];

        ret += c1_ * temp_colour_mat;
      }

      return ret;
    }

    template <typename Real, int Nc>
    Real RectangleAction<Real, Nc>::local_action(
        const typename Action<Real, Nc>::GaugeField& gauge_field,
        const Int link_index) const
    {
      // Compute the contribution to the action from the specified site
      auto staple = compute_staples(gauge_field, link_index);
      auto link = gauge_field(link_index / gauge_field.site_size(),
                              link_index % gauge_field.site_size());
      return -this->beta() * (link * staple).trace().real() / Nc;
    }
  }
}


#endif //PYQCD_GAUGE_RECTANGLE_ACTION_HPP
