#ifndef PYQCD_CONJUGATE_GRADIENT_HPP
#define PYQCD_CONJUGATE_GRADIENT_HPP
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
 * Created by Matt Spraggs on 12/02/17.
 *
 * Implementation of the conjugate gradient algorithm.
 */

#include <core/qcd_types.hpp>
#include <fermions/fermion_action.hpp>

#include "linear_algebra.hpp"
#include "solution_wrapper.hpp"


namespace pyQCD
{
  template <typename Real, int Nc>
  SolutionWrapper<Real, Nc> conjugate_gradient_unprec(
      const fermions::Action<Real, Nc>& action,
      const LatticeColourVector<Real, Nc>& rhs, const Int max_iterations,
      const Real tolerance)
  {
    typedef LatticeColourVector<Real, Nc> Fermion;

    auto& layout = rhs.layout();
    Int num_spins = rhs.site_size();

    auto hermitian_rhs = action.apply_hermiticity(rhs);

    Fermion solution(layout, ColourVector<Real, Nc>::Zero(), num_spins);

    auto r = action.apply_hermiticity(action.apply_full(solution));
    r = hermitian_rhs - r;
    auto p = r;

    Real prev_residual = dot_fermions(r, r).real();

    Int final_iterations = max_iterations;
    Real final_residual = tolerance;

    for (Int i = 0; i < max_iterations; ++i) {
      auto Ap = action.apply_hermiticity(action.apply_full(p));

      std::complex<Real> alpha = prev_residual / dot_fermions(p, Ap);

      solution += alpha * p;
      r -= alpha * Ap;

      Real current_residual = dot_fermions(r, r).real();

      if (std::sqrt(current_residual) < tolerance) {
        final_iterations = i + 1;
        final_residual = std::sqrt(current_residual);
        break;
      }

      Real beta = current_residual / prev_residual;
      p = r + beta * p;
      prev_residual = current_residual;
    }

    return SolutionWrapper<Real, Nc>(std::move(solution), final_residual,
                                     final_iterations);
  }


  template <typename Real, int Nc>
  SolutionWrapper<Real, Nc> conjugate_gradient_eoprec(
      const fermions::Action<Real, Nc>& action,
      const LatticeColourVector<Real, Nc>& rhs, const Int max_iterations,
      const Real tolerance)
  {
    using Fermion = LatticeColourVector<Real, Nc>;

    const auto& layout = rhs.layout();
    const auto volume = layout.volume();
    Int num_spins = rhs.site_size();

    Fermion hermitian_rhs = rhs;
    auto hermitian_rhs_odd_view = hermitian_rhs.segment(volume / 2, volume / 2);

    // Create preconditioned source
    hermitian_rhs_odd_view =
        hermitian_rhs_odd_view -
            action.apply_odd_even(
                action.apply_even_even_inv(rhs))
                .segment(volume / 2, volume / 2);

    hermitian_rhs = action.apply_hermiticity(hermitian_rhs);
    hermitian_rhs_odd_view = hermitian_rhs.segment(volume / 2, volume / 2);

    Fermion r(layout, num_spins),
        p(layout, ColourVector<Real, Nc>::Zero(), num_spins);

    // Invert the even sites as they're easy
    auto solution = action.apply_even_even_inv(rhs);

    // Create some views to the odd sites pertaining to the above fermions
    auto r_odd_view = r.segment(volume / 2, volume / 2);
    auto p_odd_view = p.segment(volume / 2, volume / 2);
    auto solution_odd_view = solution.segment(volume / 2, volume / 2);

    auto Ap = action.apply_hermiticity(action.apply_eoprec(solution));

    auto Ap_odd_view = Ap.segment(volume / 2, volume / 2);
    r_odd_view = hermitian_rhs_odd_view - Ap_odd_view;
    p = r;

    Real prev_residual = dot_fermions(r_odd_view, r_odd_view).real();

    Int final_iterations = max_iterations;
    Real final_residual = tolerance;

    for (Int i = 0; i < max_iterations; ++i) {
      Ap = action.apply_hermiticity(action.apply_eoprec(p));

      Ap_odd_view = Ap.segment(volume / 2, volume / 2);
      std::complex<Real> alpha = prev_residual / dot_fermions(p_odd_view,
                                                              Ap_odd_view);

      solution_odd_view = solution_odd_view + alpha * p_odd_view;
      r_odd_view = r_odd_view - alpha * Ap_odd_view;

      Real current_residual = dot_fermions(r_odd_view, r_odd_view).real();

      if (std::sqrt(current_residual) < tolerance) {
        final_iterations = i + 1;
        final_residual = std::sqrt(current_residual);
        break;
      }

      Real beta = current_residual / prev_residual;
      p = r + beta * p;
      prev_residual = current_residual;
    }

    // Reverse preconditioning preparation
    auto solution_even_view = solution.segment(0, volume / 2);
    solution_even_view =
        solution_even_view -
            action.apply_even_even_inv(
                action.apply_even_odd(solution)).segment(0, volume / 2);

    return SolutionWrapper<Real, Nc>(std::move(solution), final_residual,
                                     final_iterations);
  }
}

#endif //PYQCD_CONJUGATE_GRADIENT_HPP
