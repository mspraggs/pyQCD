from __future__ import absolute_import

from pyQCD.core import LatticeColourMatrix
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