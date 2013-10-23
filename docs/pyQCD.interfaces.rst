interfaces Package
==================

This package provides several modules that provide interfaces
between the main run script and core lattice module.

:mod:`dicts` Module
-------------------

The dicts module provides a series of dictionaries that facilitate
the conversion of various verbose properties into the numeric
values required by the functions that need them. If you are
uncertain of which value to use for a particular function
argument, this is a good place to look. The default values used
when parsing XML input files may also be found in a nested
dictionary within this file. This file is hence also useful
for looking up relevant sets of XML tags for use in the XML
input files.

.. automodule:: pyQCD.interfaces.dicts
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`io` Module
----------------

The io module provides routines for reading to and from files.
The majority of this module is devoted to the XML interface, which
parses the supplied XML file into a nested dictionary. :samp:`input`
tags are stored in a list, as there may be several, depending on the
XML input file. The interface also carries some convenience
functions:::

    from pyQCD.interfaces.io import XmlInterface

    # Load and parse the xml file
    xml = XmlInterface("examples/basic.xml")

    # Print the input xml file
    print(xml)

    # Extract the settings dictionary
    settings = xml.settings
    # Several functions refer to specific sets of settings
    simulation_settings = xml.simulation()

The remaining functions within the io module are designed to
load numpy arrays from numpy zipped archives, i.e. files with
npz extensions.

.. automodule:: pyQCD.interfaces.io
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`lattice` Module
---------------------

This sole purpose of this module is to convert non-scalar measurements
from the core lattice module, which are returned as (sometimes compound)
lists, into numpy arrays. There is a package on GitHub to incorporate
a numpy interface into the boost::python library, but as yet this hasn't
been incorporated into the core lattice module. Example usage:::

    from pyQCD.core.lattice import Lattice
    from pyQCD.interfaces.lattice import LatticeInterface
    import numpy as np

    # Create a lattice with the default options
    lattice = Lattice()
    # Since python assigns by reference, changes to the lattice
    # will be reflected in the interface
    lattice_inteface = LatticeInteface(lattice)

    # Thermalize the lattice
    lattice.thermalize(100)

    # Get the gauge field as a list of numpy matrices.
    links = lattice_interface.get_links()

.. automodule:: pyQCD.interfaces.lattice
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`measurements` Module
--------------------------

This module provides three methods to simplify extracting measurements
from the Lattice object in the run.py simulation function. Each function
accepts a dictionary specifying the measurements to be performed. The
three functions are used to create storage for the measurements,
perform and store the measurements and then save the measurements.

.. automodule:: pyQCD.interfaces.measurements
    :members:
    :undoc-members:
    :show-inheritance:

