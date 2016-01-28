#ifndef GAUGE_ACTION_HPP
#define GAUGE_ACTION_HPP

/* This file contains the base implementation of the base gauge action type,
 * upon which all gauge actions are based. */

#include <core/types.hpp>


namespace pyQCD
{
  namespace Gauge
  {
    template <typename Real, int Nc>
    class Action
    {
    public:
      typedef ColourMatrix<Real, Nc> GaugeLink;
      typedef LatticeColourMatrix<Real, Nc> GaugeField;

      Action(const Real beta, const Layout& layout) : beta_(beta)
      { }

      virtual GaugeLink compute_staples(const GaugeField& gauge_field,
                                        const Int& site_index) const = 0;

      virtual Real local_action(const GaugeField& gauge_field,
                                const Int& site_index) const = 0;

      Real beta() const
      { return beta_; }

    private:
      Real beta_; // The inverse coupling
    };
  }
}

#endif // GAUGE_ACTION_HPP