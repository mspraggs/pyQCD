from __future__ import absolute_import

import numpy as np

from pyQCD.core import LexicoLayout, LatticeColourMatrix, ColourMatrix
from pyQCD.gauge.gauge import *


def cold_start(lattice_shape):
    """Creates an SU(N) gauge field with each link set to the identity.

    Arguments:
      lattice_shape (iterable): The lattice shape the gauge field should
        exhibit.

    Returns:
      LatticeColourMatrix: A lexicographic, cold-start gauge field.
    """

    layout = LexicoLayout(lattice_shape)
    ret = LatticeColourMatrix(layout, len(lattice_shape))

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
      LatticeColourMatrix: A lexicographic, hot-start gauge field.
    """

    lattice_shape = list(lattice_shape)
    layout = LexicoLayout(lattice_shape)
    ret = LatticeColourMatrix(layout, len(lattice_shape))

    ret_view = ret.as_numpy

    for index in np.ndindex(*(lattice_shape + [len(lattice_shape)])):
        ret_view[index] = ColourMatrix.random().as_numpy

    return ret
