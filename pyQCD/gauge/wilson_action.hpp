#ifndef WILSON_ACTION_HPP
#define WILSON_ACTION_HPP

/* Here we implement the Wilson gauge action. */

#include "gauge_action.hpp"


namespace pyQCD
{
  namespace Gauge
  {
    template <typename Real, int Nc>
    class WilsonAction : public Action<Real, Nc>
    {
    public:
      WilsonAction(const Real beta, const Layout& layout);

      typename Action<Real, Nc>::GaugeLink compute_staples(
        const typename Action<Real, Nc>::GaugeField& gauge_field,
        const Int site_index) const;

      Real local_action(
        const typename Action<Real, Nc>::GaugeField& gauge_field,
        const Int site_index) const;

    private:
      std::vector<std::vector<Int>> links_;
    };

    template <typename Real, int Nc>
    WilsonAction<Real, Nc>::WilsonAction(const Real beta, const Layout& layout)
      : Action<Real, Nc>(beta, layout)
    {
    }

    template <typename Real, int Nc>
    typename Action<Real, Nc>::GaugeLink
    WilsonAction<Real, Nc>::compute_staples(
      const typename Action<Real, Nc>::GaugeField& gauge_field,
      const Int site_index) const
    {
      auto ret = Action<Real, Nc>::GaugeLink::Zero().eval();
      auto temp_colour_mat = ret;

      auto& layout = gauge_field.layout();
      Site link_coords = layout.compute_site_coords(site_index);
      auto num_dims = gauge_field.layout().num_dims() - 1;

      // Determine which plane the link does not contribute to.
      Site planes(num_dims - 1);
      int j = 0;
      for (unsigned i = 0; i < num_dims; ++i) {
        if (link_coords[num_dims] != i) {
          planes[j++] = i;
        }
      }

      for (unsigned i = 0; i < num_dims - 1; ++i) {
        std::vector<int> link_coords_copy(link_coords.size(), 0);
        std::copy(link_coords_copy.begin(), link_coords_copy.end(),
          link_coords.begin());

        /* First compute the loop above the link. Essentially we're interested
         * in:
         *
         * U_\nu (x + \mu) U^\dagger_\mu (x + \nu) U^\dagger_\nu (x)
         */
        // First link in U_\nu (x + \mu)
        link_coords_copy[num_dims] = planes[i];
        link_coords_copy[link_coords[num_dims]]++;
        layout.sanitize_site_coords(link_coords_copy);
        temp_colour_mat = gauge_field(link_coords_copy);
        // Next link is U^\dagger_\mu (x + \nu)
        link_coords_copy[num_dims] = link_coords[num_dims];
        link_coords_copy[link_coords[num_dims]]--;
        link_coords_copy[planes[i]]++;
        layout.sanitize_site_coords(link_coords_copy);
        temp_colour_mat *= gauge_field(link_coords_copy).adjoint();
        // Next link is U^\dagger_\nu (x)
        link_coords_copy[planes[i]]--;
        link_coords_copy[num_dims] = planes[i];
        layout.sanitize_site_coords(link_coords_copy);
        temp_colour_mat *= gauge_field(link_coords_copy).adjoint();

        ret += temp_colour_mat;

        /* Now we want to compute the link below the link. Essentially:
         *
         * U^\dagger _\nu (x + \mu - \nu) U^\dagger_\mu (x - \nu)
         *   U^\dagger_\nu (x - \nu)
         */
        // First link is U^\dagger_\nu (x + \mu - \nu)
        link_coords_copy[link_coords[num_dims]]++;
        link_coords_copy[planes[i]]--;
        layout.sanitize_site_coords(link_coords_copy);
        temp_colour_mat = gauge_field(link_coords_copy).adjoint();
        // Next link is U^\dagger_\mu (x - \nu)
        link_coords_copy[num_dims] = link_coords_copy[num_dims];
        link_coords_copy[link_coords[num_dims]]--;
        layout.sanitize_site_coords(link_coords_copy);
        temp_colour_mat *= gauge_field(link_coords_copy).adjoint();
        // Next link is U_\nu (x - \nu)
        link_coords_copy[num_dims] = planes[i];
        layout.sanitize_site_coords(link_coords_copy);
        temp_colour_mat *= gauge_field(link_coords_copy);

        ret += temp_colour_mat;
      }

      return ret;
    }

    template <typename Real, int Nc>
    Real WilsonAction<Real, Nc>::local_action(
      const typename Action<Real, Nc>::GaugeField& gauge_field,
      const Int site_index) const
    {
      // Compute the contribution to the action from the specified site
      auto staple = compute_staples(gauge_field, site_index);
      auto link = gauge_field(site_index);
      return -this->beta() * (link * staple).trace().real() / Nc;
    }
  }
}

#endif // WILSON_ACTION_HPP