"""Python level comms initialisation"""

from __future__ import absolute_import, division

from mpi4py import MPI
import numpy as np

from . import core


def init_comms(lattice_shape, halo_depth=1):
    """Docstring"""
    nprocs = MPI.COMM_WORLD.Get_size()
    lattice_shape = np.array(lattice_shape)
    mpi_shape = np.array(MPI.Compute_dims(nprocs, lattice_shape.size))

    comm = MPI.COMM_WORLD.Create_cart(mpi_shape, reorder=True)
    if (lattice_shape % mpi_shape).sum() > 0:
        raise RuntimeError("Invalid number of MPI processes")
    core.init_comms(comm)