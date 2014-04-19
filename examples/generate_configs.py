import pyQCD

"""
Here we generate a 4c8 ensemble with 100 configurations, using the Symanzik
rectangle-improved gauge action.
"""

if __name__ == "__main__":
    
    # Here we create the simulation object, specifying 100 configs, 10 updates
    # in between measurements and 100 updates to thermalize the lattice
    simulation = pyQCD.Simulation(100, 10, 100)

    # Now we create the lattice. Possible actions are specified in the keys of
    # pyQCD.gauge_actions. Here we set the inverse coupling, beta, to 5.5. We
    # set the lattice size to 4^3 x 8 in the first two arguments
    simulation.create_lattice(4, 8, "rectangle_improved", 5.5)

    # Now we can specify the measurements we want to do. All we want to do here
    # is store the field configurations, so we use the function
    # pyQCD.Lattice.get_config to do this, which is a member function of the
    # lattice object. This function returns the type Config, so we specify the
    # type as such here too. The resulting configs will be stored in pyQCD's
    # DataSet type, before being written to disk, so we need to specify a
    # file name for the output. It is also possible to specify a message that
    # is outputted when the measurement takes place. Here we specify one.
    simulation.add_measurement(pyQCD.Lattice.get_config, pyQCD.Config,
                               "4c8_ensemble.zip",
                               meas_message="Saving current config")
    
    # Now that the lattice and measurements have been set up, the simulation can
    # be run
    simulation.run()

    # Once finished, there should be a zip archive called 4c8_ensemble.zip in the
    # examples directory, which should contain 100 configurations.
    # In addition, since the default options were used when calling run(), the
    # simulation object would, at the end of the simulation, contain a member
    # variable called plaquettes, containing the mean plaquette value for each of
    # the configurations. This could optionally be saved to disk as a numpy
    # binary.
