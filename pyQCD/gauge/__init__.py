from __future__ import absolute_import

import numpy as np

from pyQCD.core import LatticeColourMatrix, random_colour_matrix
from pyQCD.gauge.gauge import *


def cold_start(lattice_shape):
    """Creates an SU(N) gauge field with each link set to the identity.

    Arguments:
      lattice_shape (iterable): The lattice shape the gauge field should
        exhibit.
    """
    lattice_shape = list(lattice_shape)
    ret = LatticeColourMatrix(lattice_shape, len(lattice_shape))

    ret_view = ret.as_numpy
    num_colours = ret_view.shape[-1]

    for i in range(num_colours):
        ret_view[..., i, i] = 1.0

    return ret

def hot_start(lattice_shape):
    """Creates an SU(N) gauge field with random links.

    Arguments:
      lattice_shape (iterable): The lattice shape the gauge field should
        exhibit.

    Returns:
      LatticeColourMatrix: The hot-start gauge field.
    """
    # TODO: Move the below into Cython/C++ code for speed

    lattice_shape = list(lattice_shape)
    ret = LatticeColourMatrix(lattice_shape, len(lattice_shape))

    ret_view = ret.as_numpy

    for index in np.ndindex(*(lattice_shape + [len(lattice_shape)])):
        ret_view[index] = random_colour_matrix().as_numpy

    return ret
