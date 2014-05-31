from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import time
import logging

import numpy as np

from .log import _logger

class Simulation(object):
    """Creates, configures and rns a lattice simulation
    
    Before being able to run a simulation, a Lattice object must be created and
    passed to the Simulation object. A gauge ensemble can then specified,
    the configurations of which will then be loaded into the Lattice object
    during the simulation. If no ensemble is specified, then a gauge field will
    be initialized and thermalized prior to performing measurements.
    Measurements are then specified by passing functions to the Simulation
    object.

    Attributes:
      lattice (Lattice): The lattice object used by the simulation.
        measurements.
      measurements (list): Specifies the mesurements to be performed on the
        gauge configuration, in the form of a list of lists, with each sub-list
        specifying the measurement in the following format:
        [measurement function, measurement_message, function_keyword_args,
         dataset_object_containing_measurements].
      num_configs (int): The number of configurations on which measurements
        are performed.
      num_warmup_updates (int): The number of updates used to thermalize the
        lattice

    Args:
      lattice (Lattice): The lattice object to use during the simulation.
      num_configs (int): The number of configurations on which to perform
        measurements.
      num_warmup_updates (int): The number of updates used to thermalize the
        lattice.
        
    Examples:
      Create a simulation to perform measurements on 100 configurations,
      performing the meausrements every 10 updates. The lattice will be
      thermalized by doing 100 updates, so 1100 updates are performed in
      total. The simulation defaults to using heatbath updates and partitions
      the lattice, parallelising each update. The random number generator is
      fed a random seed and the verbosity level is set at medium.
      
      >>> import pyQCD
      >>> simulation = pyQCD.Simulation(100, 10, 100)
      
      Create a lattice object before adding it to  a simulation. Then add the
      get_config function (a member function of pyQCD.Lattice). The function
      returns the current field configuration as a numpy ndarray, and we store
      these in "4c8_wilson_purgaug_configs".
      
      >>> import pyQCD
      >>> lattice = pyQCD.Lattice(4, 8, 5.5, "wilson", 10)
      >>> sim = pyQCD.Simulation(lattice, 100, 100)
      >>> fname = "4c8_wilson_purgaug_configs.zip"
      >>> sim.add_measurement(pyQCD.Lattice.get_config,
      ...                     pyQCD.io.write_datum_callback(fname))
      >>> sim.run()
    """

    def __init__(self, lattice, num_configs, num_warmup_updates):
        """Constructor for pyQCD.Simulation (see help(pyQCD.Simulation))"""

        self.num_configs = num_configs
        self.num_warmup_updates = num_warmup_updates
        self.lattice = lattice
        self.measurements = []

        self.use_ensemble = False

        self.logger = logging.getLogger("pyQCD.simulation")

    def specify_ensemble(self, load_func, indices=None):
        """Directs the simulation object to use the specified gauge ensemble

        Args:
          load_func (function): The function to load the specified field
            configuration. Must accept an integer specifying the configuration
            number as its first argument and return a numpy ndarray with shape
            (T, L, L, L, 4, 3, 3).
          indices (list, optional): The config numbers to load. Must have length
            equal to the value in the simulation num_configs member variable.

        Examples:
          Here we create a simulation and specify an ensemble of measurements
          contained within a zip archive, with each configuration in numpy
          binary format. Note that the second argument of specify_ensemble in
          this case is actually the same as what would have been assumed by
          default. As no measurememts have been specified in this simulation,
          when run is called, very little happens.

          >>> import pyQCD
          >>> lattice = pyQCD.Lattice(4, 8, 5.5, "wilson", 10)
          >>> sim = pyQCD.Simulation(lattice, 100, 100)
          >>> fname = "ensemble.zip"
          >>> sim.specify_ensemble(pyQCD.io.extract_datum_callback(fname),
          ...                      list(range(100)))
          >>> sim.run()
        """
        
        self.config_loader = load_func
        self.ensemble_indices = (list(range(self.num_configs))
                                 if indices == None
                                 else indices)
        self.use_ensemble = True

    def add_measurement(self, meas_func, callback, args=(), kwargs={}):
        """Adds a measurement to the simulation nto be performed when the
        simulation is run

        Args:
          meas_func (function): The function defining the measurement. This
            must accept a lattice object as the first argument.
          callback (function): A function used to further process the return
            value from meas_func. For example this could typically be used
            to save the measurement result to disk in some way. The function
            must accept exactly two arguments - the first the return value of
            meas_func and the second an integer signifying the index of the
            current config (which varies from 0 to self.num_configs - 1).
          args (list, optional): The additional arguments to supply to
            the measurement function.
          kwargs (dict, optional): The additional keyword arguments to
            supply to the measurement function.

        Examples:
          Create a lattice, pass it to a new simulation object, then add
          a measurement to save the current field configuration to disk. Here
          we use the pyQCD io function write_datum_callback, which generates
          a function that wraps pyQCD.io.write_datum.

          >>> import pyQCD
          >>> lattice = pyQCD.Lattice(4, 8, 5.5, "wilson", 10)
          >>> sim = pyQCD.Simulation(lattice, 100, 100)
          >>> sim.add_measurement(pyQCD.Lattice.get_config,
          ...                     pyQCD.io.write_datum_callback("ensemble.zip"))
          >>> sim.run()
        """

        self.measurements.append((meas_func, callback, args, kwargs))

    def run(self):
        """Runs the simulation, including any added measurements"""

        self.plaquettes = np.zeros(self.num_configs)

        self._log_settings()

        logger = _logger()

        if not self.use_ensemble:
            logger.info("Thermalizing lattice")
            self.lattice.thermalize(self.num_warmup_updates)

        for i in range(self.num_configs):
            logger.info("Configuration: {}".format(i))

            if self.use_ensemble:
                config = self.config_loader(self.ensemble_indices[i])
                self.lattice.set_config(config)

            else:
                self.lattice.next_config()

            self.plaquettes[i] = self.lattice.get_av_plaquette()

            for meas, callback, args, kwargs in self.measurements:
                result = meas(self.lattice, *args, **kwargs)
                callback(result, i)

        logger.info("Simulation complete")

    def _log_settings(self):
        """Spits out the settings for the simulation"""

        logger = _logger()
        
        logger.info("Running measuremements on {} configurations"
                    .format(self.num_configs))
        if not self.use_ensemble:
            logger.info("Measurement frequency: {} configurations"
                        .format(self.lattice.num_cor))

        logger.info("Lattice shape: {}".format(self.lattice.shape))
        if not self.use_ensemble:
            logger.info("Gauge action: {}".format(self.lattice.action))
            logger.info("Inverse coupling (beta): {}"
                        .format(self.lattice.beta))

        logger.info("Mean temporal link (ut): {}".format(self.lattice.ut))
        logger.info("Mean spatial link (us): {}".format(self.lattice.us))
        logger.info("Anisotropy factor (chi): {}".format(self.lattice.chi))

        if not self.use_ensemble:
            logger.info("Parallel sub-lattice size: {}"
                        .format(self.lattice.block_size))

        for meas, callback, args, kwargs in self.measurements:
            messages = ["Settings for measurement function {}\n"
                        .format(meas.__name__)]
            messages.extend(["  {}: {}\n".format(name, val)
                             for name, val in zip(meas.__code__.co_varnames,
                                                  args)])
            messages.extend(["  {}: {}\n".format(name, val)
                             for name, val in kwargs.items()])

            logger.info("".join(messages))
