from lattice import Lattice
from config import Config
from propagator import Propagator
from wilslps import WilsonLoops
from dataset import DataSet

import numpy as np
import sys
import time
import inspect
import warnings

class Simulation(object):
    """Creates and returns a simulation object
    
    Args:
      num_configs (int): The number of configurations on which to perform
        measurements
      measurement_spacing (int): The spacing of configurations on which
        measurements are performed
      num_warmup_updates (int): The number of updates used to thermalize the
        lattice
      update_method (str): The algorirthm to use when updating the gauge
        field configuration. Currently "heatbath", "metropolis" and
        "staple_metropolis" are supported (see help(pyQCD.Lattice) for details)
      run_parallel (bool): Determines whether to partition the
        lattice into a red-black/checkerboard pattern, then perform
        updates on the red blocks in parallel before updating the
        black blocks. (Requires OpenMP to work.)
      rand_seed (int): The seed to be used by the random number
        generator (-1 specifies that a seed based on the time should
        be used).
      verbosity (int): The level of verbosity when performing the simulation,
        with 0 producing no output, 1 producing some output and 2 producing
        the most output, suc as details of propagator inversions.
        
    Examples:
      Create a simulation to perform measurements on 100 configurations,
      performing the meausrements every 10 updates. The lattice will be
      thermalized by doing 100 updates, so 1100 updates are performed in
      total. The simulation defaults to using heatbath updates and partitions
      the lattice, parallelising each update. The random number generator is
      fed a random seed and the verbosity level is set at medium.
      
      >>> import pyQCD
      >>> simulation = pyQCD.Simulation(100, 10, 100)
      """
    
    xml_meas_dict = {"propagator": (Propagator, "get_propagator"),
                     "wilson_loops": (WilsonLoops, "get_wilson_loops"),
                     "configuration": (Config, "get_config")}
    
    def __init__(self, num_configs, measurement_spacing, num_warmup_updates,
                 update_method="heatbath", run_parallel=True, rand_seed=-1,
                 verbosity=1):
        """Constructor for pyQCD.Simulation (see help(pyQCD.Simulation))"""
        
        self.num_configs = num_configs
        self.measurement_spacing = measurement_spacing
        self.num_warmup_updates = num_warmup_updates
        
        self.update_method = update_method
        self.run_parallel = run_parallel
        self.rand_seed = rand_seed
        self.verbosity = verbosity
        
        self.use_ensemble = False
        
        self.measurements = []
    
    def create_lattice(self, L, T, action, beta, u0=1.0, block_size=None):
        """Creates a Lattice instance to use in the simulation
        
        Args:
          L (int): The lattice spatial extent
          T (int): The lattice temporal extent
          action (str): The gauge action to use in gauge field updates. For
            additional details see help(pyQCD.Lattice).
          beta (float): The inverse coupling to use in the gauge action.
          u0 (float): The mean link/tadpole improvement parameter
          block_size (int): The side length of the sub-lattices used to
            partition the lattice for parallel updates. If left as None, then
            this will be determined automatically.
            
        Examples:
          Create a simulation and add a lattice to it, using the Symanzik
          rectangle-improved gauge action, an inverse coupling of 4.26 and
          a mean link of 0.852 (this should produce a Sommer scale lattice
          spacing of ~0.25 fm).
          
          >>> import pyQCD
          >>> simulation = pyQCD.Simulation(100, 10, 100)
          >>> simulation.add_lattice(4, 8, "rectangle_improved", 4.26, 0.852)
        """
        
        self.lattice = Lattice(L, T, beta, u0, action, self.measurement_spacing,
                               self.update_method, self.run_parallel,
                               block_size, self.rand_seed)
    
    def load_ensemble(self, filename):
        """Loads a gauge configuration dataset to use in the simulation
        
        Args:
          filename (str): The name of the file containing the ensemble
        
        Raises:
          AttributeError: If the number of configurations in the ensemble
            do not match the number of configurations in the simulation, or
            if the lattice extents in the ensemble do not match those of the
            lattice object in the simulation.
            
        Examples:
          Create a simulation to measure on 100 configurations, with each
          measurement separated by 10 updates. Here the number of warmup
          updates means nothing, so we set it to None. Then we create a
          4^3 x 8 lattice using the Wilson action, before loading an
          existing ensemble. The ensembly must contain the same number of
          configs as entered when creating the Simulation object, and the
          extents of the lattice must be the same as those entered when
          creating the lattice.
          
          >>> import pyQCD
          >>> simulation = pyQCD.Simulation(100, 10, None)
          >>> simulation.create_lattice(8, 4, "wilson", 5.5)
          >>> simulation.load_ensemble("my_configs.zip")
        """
        
        if not hasattr(self, "lattice"):
            raise AttributeError("A lattice must be defined before an ensemble "
                                 "may be loaded.")
        
        ensemble = DataSet.load(filename)
        
        if ensemble.num_data != self.num_configs:
            raise AttributeError("Number of configutations in ensemble ({}) "
                                 "does not match the required number of "
                                 "simulation configurations ({})."
                                 .format(ensemble.num_data, self.num_configs))
        elif self.lattice.L != ensemble.get_datum(0).L:
            raise AttributeError("Ensemble spatial extent ({}) does not match "
                                 "the specified lattice spatial extent ({})."
                                 .format(ensemble.get_datum(0).L,
                                         self.lattice.L))
        elif self.lattice.T != ensemble.get_datum(0).T:
            raise AttributeError("Ensemble temporal extent ({}) does not match "
                                 "the specified lattice temporal extent ({})."
                                 .format(ensemble.get_datum(0).T,
                                         self.lattice.T))
        else:
            self.ensemble = ensemble
            self.use_ensemble = True
            
    def add_measurement(self, meas_function, meas_type, meas_file, kwargs={},
                        meas_message=None):
        """Adds a measurement to the simulation to be performed when the
        simulation is run
        
        Args:
          meas_function (function): The function defining the measurement.
            This must accept a lattice object as the first argument, and
            return a type specified my meas_type.
          meas_type (type): The type of the object returned by
            meas_function.
          meas_file (str): The name of the file in which the measurements
            will be stored.
          kwargs (dict, optional): The additional keyword arguments to
            supply to the measurement function.
          meas_message (str, optional): The message to display when
            the measurement is being performed, if verbose output is
            turned on. If this argument is None, the function name is
            displayed.
            
        Examples:
          Create a simulation object, create a lattice and add the
          get_propagator function (a member function of pyQCD.Lattice).
          The function returns a pyQCD.Propagator object, and we store
          these in "4c8_wilson_purgaug_propagators".
          
          When run, the simulation should print "Running get_propagator"
          
          >>> import pyQCD
          >>> sim = pyQCD.Simulation(100, 10, 100)
          >>> sim.create_lattice(4, 8, "wilson", 5.5)
          >>> sim.add_measurement(pyQCD.Lattice.get_propagator,
          ...                     pyQCD.Propagator,
          ...                     "4c8_wilson_purgaug_propagators.zip"
          ...                     {"mass": 0.4})
        """
        
        if "verbosity" in meas_function.func_code.co_varnames \
          and not "verbosity" in kwargs.keys():
            kwargs.update({"verbosity": self.verbosity})
        
        if meas_message == None:
            meas_message = "Running {}".format(meas_function.__name__)

        self.measurements.append([meas_function, meas_message, kwargs,
                                  DataSet(meas_type, meas_file)])
    
    def _do_measurements(self, save=True):
        """Iterate through self.measurements and gather results"""
        
        for measurement in self.measurements:
            if self.verbosity > 0:
                print("- {}...".format(measurement[1]))
                sys.stdout.flush()
                
            meas_result = measurement[0](self.lattice, **measurement[2])
                
            if save:
                measurement[3].add_datum(meas_result)
                
            if self.verbosity > 0:
                print("Done!")
                sys.stdout.flush()
        
    def run(self, timing_run=False, num_timing_configs=10, store_plaquette=True):
        """Runs the simulation
        
        Args:
          timing_run (bool, optional): Determines whether this is a test
            run. If True, the simulation perform the number of updates
            specified by num_timing_configs. No measurements will be saved,
            and an estimated time for the full simulation to complete
            will be printed.
          num_timing_configs (int, optional): Determines the number of
            configurations generated if a timing run is being performed.
          store_plaquette (bool, optional): Determines whether the average
            plaquette value for the lattice should be stored after each
            update. If this is True, then the plaquettes will be stored in
            the member variable plaquettes, which has type numpy.ndarray.
            
        Examples:
          Create a simulation object, create a lattice and add the
          get_config function (a member function of pyQCD.Lattice).
          The function returns a pyQCD.Config object, and we store
          these in "4c8_wilson_purgaug_configs".
          
          >>> import pyQCD
          >>> sim = pyQCD.Simulation(100, 10, 100)
          >>> sim.create_lattice(4, 8, "wilson", 5.5)
          >>> sim.add_measurement(pyQCD.Lattice.get_config,
          ...                   pyQCD.Config,
          ...                   "4c8_wilson_purgaug_configs.zip")
          >>> sim.run()
          Simulation Settings
          -------------------
          Number of configurations: 100
          Measurement spacing: 10
          Thermalization updates: 100
          Update method: heatbath
          Use OpenMP: True
          Random number generator seed: -1
          # Blank line
          Lattice Settings
          ----------------
          Spatial extent: 4
          Temporal extent: 8
          Gauge action: wilson
          Inverse coupling (beta): 5.5
          Mean link (u0): 1.0
          Parallel sub-lattice size: 4
          # Blank line
          Get Config Measurement Settings
          -------------------------------
          Filename: /absolute/path/to/configs.zip
          # Blank line
          Thermalizing lattice...  Done!
          Configuration: 0
          Updating gauge field...  Done!
          Average plaquette: 0.499948844134
          Performing measurements...
          - Running get_config...
          Done!
          .
          .
          .
          Simulation completed in 0 hours, 5 minutes and 16.9606249332 seconds
          # Blank line
          >>> simulation.plaquettes.shape
          (100,)
          """
        
        if store_plaquette:
            self.plaquettes = np.zeros(self.num_configs)
            
        if self.verbosity > 0:
            print(self)
        
        t0 = time.time()
        
        if not self.use_ensemble:
            if self.verbosity > 0:
                print("Thermalizing lattice..."),
                sys.stdout.flush()
        
            self.lattice.thermalize(self.num_warmup_updates)
        
            if self.verbosity > 0:
                print(" Done!")
                
        if timing_run:
            N = num_timing_configs
        else:
            N = self.num_configs
            
        t1 = time.time()
            
        for i in xrange(N):
            if self.verbosity > 0:
                print("Configuration: {}".format(i))
                sys.stdout.flush()
            
            if self.use_ensemble:
                if self.verbosity > 0:
                    print("Loading gauge field..."),
                    sys.stdout.flush()
                    
                config = self.ensemble.get_datum(i)
                self.lattice.set_config(config)
                
                if self.verbosity > 0:
                    print(" Done!")
            
            else:
                if self.verbosity > 0:
                    print("Updating gauge field..."),
                    sys.stdout.flush()
                    
                self.lattice.next_config()
                
                if self.verbosity > 0:
                    print(" Done!")
                    
            if store_plaquette:
                self.plaquettes[i] = self.lattice.get_av_plaquette()
                if self.verbosity > 0:
                    print("Average plaquette: {}"
                          .format(self.plaquettes[i]))
                    
            if self.verbosity > 0:
                print("Performing measurements...")
                sys.stdout.flush()
            self._do_measurements(not timing_run)
            
            if self.verbosity > 0:
                print("")
            
        t2 = time.time()
        
        if self.verbosity > 0:
        
            total_time = (t2 - t1) / N * self.num_configs + t1 - t0 \
              if timing_run else t2 - t0
          
            hrs = int((total_time) / 3600)
            mins = int((total_time - 3600 * hrs) / 60)
            secs = total_time - 3600 * hrs - 60 * mins
    
            if timing_run:
                print("Estimated run time: {} hours, {} minutes and {} seconds"
                      .format(hrs, mins, secs))
            else:
                print("Simulation completed in {} hours, {} minutes and {} "
                      "seconds".format(hrs, mins, secs))
    
    def __str__(self):
        
        out = \
          "Simulation Settings\n" \
          "-------------------\n" \
          "Number of configurations: {}\n" \
          "Measurement spacing: {}\n" \
          "Thermalization updates: {}\n" \
          "Update method: {}\n" \
          "Use OpenMP: {}\n" \
          "Random number generator seed: {}\n" \
          "\n" \
          "Lattice Settings\n" \
          "----------------\n" \
          "Spatial extent: {}\n" \
          "Temporal extent: {}\n" \
          "Gauge action: {}\n" \
          "Inverse coupling (beta): {}\n" \
          "Mean link (u0): {}\n" \
          "Parallel sub-lattice size: {}\n" \
          "\n".format(self.num_configs, self.measurement_spacing,
                      self.num_warmup_updates, self.update_method,
                      self.run_parallel, self.rand_seed, self.lattice.L,
                      self.lattice.T, self.lattice.action, self.lattice.beta,
                      self.lattice.u0, self.lattice.block_size)
        
        if len(self.measurements) > 0:
            for measurement in self.measurements:
                heading_underline \
                  = (len(measurement[0].__name__) + 21) * "-"
                meas_settings = \
                  "{} Measurement Settings\n" \
                  "{}\n".format(measurement[0].__name__.replace("_", " ")
                                .title(),
                                heading_underline)
            
                meas_settings \
                  = "".join([meas_settings,
                             "Filename: {}\n".format(measurement[3].filename)])
                
                for key, value in measurement[2].items():
                    meas_settings = "".join([meas_settings,
                                        "{}: {}\n".format(key, value)])
                
        
            out = "".join([out, meas_settings])
            
        return out
