core Package
============

The core package contains native binary modules that perform
operations that are required to be fast, such as gauge field
updates and propagator computations. Currently this is
achieved through the use of the Lattice object within the
lattice module.

:mod:`lattice` Module
---------------------

The lattice module contains the Lattice class, which encodes
all information pertaining to the gauge field, and performs
most of the more numerically intensive computations. The
module is a C++ binary, wrapped using boost::python.

The Lattice object constructor allows the properties of the
lattice to be defined, namely the spatial and temporal
extents of the lattice. The gauge action and its associated
parameters are also defined in the constructor. Finally,
parameters pertaining to the simulation are defined, such as
the number of gauge configurations between measurements, the
method used to update the gauge field and whether the lattice
is updated in parallel or not.

The object-orientated nature of this module is designed to
facilitate many of the operations required in a typical
lattice QCD simuation. The properties of the lattice may be
defined, then the gauge field updated and measurements
performed on the gauge field. For example:::

    from pyQCD.core.lattice import Lattice
    import numpy as np

    # Create the lattice object. Wilson's gauge action by default,
    # with ten configurations between each measurement.
    # By default, the lattice is initialised cold, meaning all links
    # are unit matrices.
    lattice = Lattice(L=8, T=16, beta=6.0)

    # Gather 1000 average plaquette values
    n_configs = 1000
    plaquettes = np.zeros(n_configs)

    for i in xrange(n_configs):
        plaquettes[i] = lattice.av_plaquette()
        lattice.update()

Similarly more complex measurements may also be made:::

    from pyQCD.core.lattice import Lattice
    import numpy as np

    # This time go for Symanzik's rectangle improved Wilson action
    lattice = Lattice(L=8, T=16, beta=6.0, action=1)
    # Thermalize the lattice by doing 100 updates
    lattice.thermalize(100)

    # We're going to get the propagators for 100 configurations.
    # There are 8192 lattice sites, and the propagator to each site
    # is a 12-by-12 matrix, hence the shape of the numpy array.
    propagators = np.zeros((100, 8192, 12, 12))

    for i in xrange(100):
        # Get the propagator for a quark with a mass of 0.4 in
        # lattice units.
        propagators[i] = np.array(lattice.propagator(0.4))
        # next_config updates the lattice Ncor times (10 in this case).
	lattice.next_config()

Within the C++ code, the gauge field takes the form of a
:samp:`std::vector` of complex 3-by-3 matrices. The gauge
field has hence been flattened to accomodate a gauge field
with one spin index and four space-time coordinates within a
one-dimensional vector. Given a link lying in axis :samp:`mu`
at the lattice site given by coordinates :samp:`t`, :samp:`x`,
:samp:`y` and :samp:`z`, the index of this link within the
:samp:`std::vector` is given by::

    index = mu + 4 * z + 4 * L * y + 4 * L**2 * x + 4 * L**3 * t

.. automodule:: pyQCD.core.lattice
    :members:
    :undoc-members:
    :show-inheritance:

