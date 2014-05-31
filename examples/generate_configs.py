"""
Here we generate a 4c8 ensemble with 100 configurations, using the Symanzik
rectangle-improved gauge action.
"""

import logging

import pyQCD

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)

    # First we create the lattice. Possible actions are specified in the keys of
    # pyQCD.gauge_actions. Here we set the inverse coupling, beta, to 5.5. We
    # set the lattice size to 4^3 x 8 in the first two arguments. The spacing
    # between measurements is set to ten updates.
    lattice = pyQCD.Lattice(4, 8, 5.5, "rectangle_improved", 10)

    # Here we create the simulation object, specifying 100 configs an 100
    # thermalization updates
    simulation = pyQCD.Simulation(lattice, 100, 100)

    # Now we can specify the measurements we want to do. All we want to do here
    # is store the field configurations, so we use the function
    # pyQCD.Lattice.get_config to do this, which is a member function of the
    # lattice object. After a measurement is performed, a callback function is
    # called to process the result of the measurement in some way. Here we
    # create a callback function based on the write_datum function in the io
    # submodule. This saves the result as a numpy binary and adds the file to
    # a zip archive.
    fname = "4c8_ensemble.zip"
    simulation.add_measurement(pyQCD.Lattice.get_config,
                               pyQCD.io.write_datum_callback(fname))
    
    simulation.run()

    # Once finished, there should be a zip archive called 4c8_ensemble.zip in
    # the examples directory, which should contain 100 configurations.
    # In addition, since the default options were used when calling run(), the
    # simulation object would, at the end of the simulation, contain a member
    # variable called plaquettes, containing the mean plaquette value for each
    # of the configurations. This could optionally be saved to disk as a numpy
    # binary.
