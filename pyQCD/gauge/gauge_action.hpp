#ifndef GAUGE_ACTION_HPP
#define GAUGE_ACTION_HPP

/* This file contains the base implementation of the base gauge action type,
 * upon which all gauge actions are based. */

#include <core/types.hpp>


namespace pyQCD {

  template <typename Real, int Nc>
  class GaugeAction
  {
  public:
    GaugeAction(const Real beta) : beta_(beta) { }
  
    virtual ColourMatrix<Real, Nc> compute_staples(
      const LatticeColourMatrix<Real, Nc>& gauge_field,
      const Int& site_index) const = 0;

    virtual Real local_action(
      const LatticeColourMatrix<Real, Nc>& gauge_field,
      const Int& site_index) const = 0;

    Real beta() const { return beta_; }

  private:
    Real beta_; // The inverse coupling
  };

}

#endif // GAUGE_ACTION_HPP