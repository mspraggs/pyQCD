from __future__ import absolute_import

import numpy as np

from pyQCD.core.core import *


def random_colour_matrix():
    """Generate a ColourMatrix instance belonging to SU(N).

    Returns:
      ColourMatrix: Belonging to SU(N)
    """

    ret = ColourMatrix()
    ret_view = ret.as_numpy
    num_colours = ret_view.shape[-1]

    new_vec = ((2 * np.random.random(num_colours) - 1) +
               (2 * np.random.random(num_colours) - 1) * 1j)
    new_vec /= np.vdot(new_vec, new_vec)**0.5

    ret_view[0] = new_vec

    for i in range(1, num_colours):
        new_vec = ((2 * np.random.random(num_colours) - 1) +
                   (2 * np.random.random(num_colours) - 1) * 1j)

        for j in range(i):
            dot = np.vdot(ret_view[j], new_vec)
            new_vec -= dot * ret_view[j]
            assert np.allclose(np.vdot(new_vec, ret_view[j]), 0.0)

        new_vec /= np.vdot(new_vec, new_vec)**0.5
        ret_view[i] = new_vec

    det = np.linalg.det(ret_view)
    ret_view /= det**(1.0 / num_colours)

    return ret
