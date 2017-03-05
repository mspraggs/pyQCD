#ifndef PYQCD_SOLUTION_WRAPPER_HPP
#define PYQCD_SOLUTION_WRAPPER_HPP
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
 * Created by Matt Spraggs on 18/02/17.
 *
 * Wrapper for solution produced by iterative solver algorithms.
 */

#include <core/qcd_types.hpp>


namespace pyQCD
{
  template <typename Real, int Nc>
  class SolutionWrapper
  {
  public:
    SolutionWrapper(LatticeColourVector<Real, Nc>&& solution,
                    const Real tolerance, const Int num_iterations)
      : solution_(solution), tolerance_(tolerance),
        num_iterations_(num_iterations)
    { }
    
    const LatticeColourVector<Real, Nc>& solution() const { return solution_; }
    Real tolerance() const { return tolerance_; }
    Int num_iterations() const { return num_iterations_; }
    
  private:
    LatticeColourVector<Real, Nc> solution_;
    Real tolerance_;
    Int num_iterations_;
  };
}

#endif //PYQCD_SOLUTION_WRAPPER_HPP
