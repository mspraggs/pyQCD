"""
Provides a suite of classes with which to perform lattice simulations
and analyze results. There are two sides to the package: simulation
and analysis.

The Simulation and Lattice classes supply the majority of the routines
to run a lattice simulation.

The remaining classes serve to encapsulate data produced by simulations,
as well as providing certain routines for analysis. With the exception
of the DataSet class, these classes all inherit from the Observable base
class. The DataSet class serves to contain a series of objects that
inherit from the Observable class, as well as providing statistical
functions such as bootstrap and jackknife.

Note that in general, despite being set in 4D Euclidean space, here zero is
used to denote the time axis, and all numpy arrays use the the first
index as the index for time. In addition, the chiral representation of the
gamma matrices is used.

Attributes:
  baryons_degenerate (list): The CHROMA baryon labels for baryons with
    degenerate quark masses.
  baryons_m1m2 (list): The CHROMA baryon labels for baryons where the first
    propagator quark mass is less than the second propagator quark mass.
  baryons_m2m1 (list): The CHROMA baryon labels for baryons where the second
    propagator quark mass is less than the second propagator quark mass.
  gamma0 (numpy.ndarray): The zeroth Dirac matrix.
  gamma1 (numpy.ndarray): The first Dirac matrix.
  gamma2 (numpy.ndarray): The second Dirac matrix.
  gamma3 (numpy.ndarray): The third Dirac matrix.
  gamma4 (numpy.ndarray): The fourth Dirac matrix (equal to gamma0).
  gamma5 (numpy.ndarray): The fifth Dirac matrix.
  gammas (list): The above Dirac gamma matrices as a list of numpy arrays
  Gammas (dict): The complete basis of 16 gamma matrices. These can
    be referred to in one of three ways, following the CHROMA convention
    described here:
    http://qcdsfware.uni-r.de/chroma-tutorial/chroma_gamma_matrices.pdf
    See pyQCD.interpolators and pyQCD.mesons for lists of possible keys
    for this dictionary.
  id4 (numpy.ndarray): The 4x4 identity matrix.
  interpolators (list): The labels for the 16 possible gamma matrix
    combinations (used as keys for the Gammas dictionary).
  mesons (list): The meson labels for the 16 meson interpolators (used
    as keys for the Gammas dictionary).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from .core import *
from . import io
