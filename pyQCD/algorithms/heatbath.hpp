#ifndef HEATBATH_HPP
#define HEATBATH_HPP

/* This file contains the functions necessary to update a single gauge link
 * using the pseudo heatbath algorithm. */

#include <random>

#include <core/types.hpp>
#include <utils/matrices.hpp>


namespace pyQCD {

  SU2Matrix gen_heatbath_su2(const Real weight, const Real beta);

  void heatbath_update(LatticeColourMatrix& gauge_field,
                       const LatticeColourMatrix& gauge_action,
                       const Int site_index);
}

#endif