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
 * Created by Matt Spraggs on 05/02/17.
 *
 *
 * Implementation of matrix utility functions.
 */

#include "matrices.hpp"


namespace pyQCD
{
  std::vector<Eigen::MatrixXcd> generate_gamma_matrices(const int num_dims)
  {
    using Mat = SpinMatrix<double>;
    int mat_size = static_cast<int>(std::pow(2.0, num_dims / 2));

    std::vector<Mat> ret(num_dims, Mat::Zero(mat_size, mat_size));

    if (num_dims <= 3) {
      ret[0] = sigma1;
      ret[1] = sigma2;

      if (num_dims == 3) {
        ret[2] = sigma3;
      }
    }
    else if (num_dims % 2 == 0) {
      auto sub_matrices = generate_gamma_matrices(num_dims - 1);

      for (int i = 1; i < num_dims; ++i) {
        ret[i].block(0, mat_size / 2, mat_size / 2, mat_size / 2)
            = -I * sub_matrices[i - 1];
        ret[i].block(mat_size / 2, 0, mat_size / 2, mat_size / 2)
            = I * sub_matrices[i - 1];
      }
      ret.front().block(0, mat_size / 2, mat_size / 2, mat_size / 2)
          = Mat::Identity(mat_size / 2, mat_size / 2);
      ret.front().block(mat_size / 2, 0, mat_size / 2, mat_size / 2)
          = Mat::Identity(mat_size / 2, mat_size / 2);
    }
    else {
      auto sub_matrices = generate_gamma_matrices(num_dims - 1);
      std::copy(sub_matrices.begin(), sub_matrices.end(), ret.begin());
      ret.back().block(0, 0, mat_size / 2, mat_size / 2)
          = Mat::Identity(mat_size / 2, mat_size / 2);
      ret.back().block(mat_size / 2, mat_size / 2, mat_size / 2, mat_size / 2)
          = -Mat::Identity(mat_size / 2, mat_size / 2);
    }
    return ret;
  }
}