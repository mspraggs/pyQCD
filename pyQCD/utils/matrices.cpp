#include "matrices.hpp"

namespace pyQCD
{
  void compute_su2_subgroup_pos(const unsigned int index,
                                unsigned int& i, unsigned int& j)
  {
    pyQCDassert((assert (index < num_colours)),
      std::range_error("SU(2) subgroup index invalid"));

    unsigned int tmp = index;
    for (i = 0; tmp >= 3 - 1 - i; ++i) {
      tmp -= (num_colours - 1 - i);
    }
    j = i + 1 + tmp;
  }

  SU2Matrix extract_su2(const ColourMatrix& colour_matrix,
                        const unsigned int subgroup)
  {
    SU2Matrix ret;
    unsigned int i, j;
    compute_su2_subgroup_pos(subgroup, i, j);

    ret(0, 0) = colour_matrix(i, i);
    ret(0, 1) = colour_matrix(i, j);
    ret(1, 0) = colour_matrix(j, i);
    ret(1, 1) = colour_matrix(j, j);

    return ret;
  }

  ColourMatrix insert_su2(const SU2Matrix& su2_matrix,
                          const unsigned int subgroup)
  {
    ColourMatrix ret;
    unsigned int i, j;
    compute_su2_subgroup_pos(subgroup, i, j);

    ret(i, i) = su2_matrix(0, 0);
    ret(i, j) = su2_matrix(0, 1);
    ret(j, i) = su2_matrix(1, 0);
    ret(j, j) = su2_matrix(1, 1);

    return ret;
  }
}