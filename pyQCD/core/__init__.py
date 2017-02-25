"""The core module provides data types with colour indices.

There are four basic types:

- ColourVector
- ColourMatrix
- LatticeColourVector
- LatticeColourMatrix

The first two types encode individual complex-valued SU(Nc) vectors and
matrices, where Nc is the number of colours.

The latter two, as their names' suggest, encode one or more of the former two
types at each site of a hypercubic lattice.

All four types implement the python buffer interface, allowing the underlying
data to be accessed directly using an indexing scheme identical to that used
in the numpy package. In addition, the various attributes of numpy arrays are
also exposed, allowing properties such as the shape and size of these objects
to be examined.
"""

from __future__ import absolute_import

from pyQCD.core.core import *
