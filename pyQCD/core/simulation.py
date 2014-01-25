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
    
    xml_meas_dict = {"propagator": (Propagator, "get_propagator"),
                     "wilson_loops": (WilsonLoops, "get_wilson_loops"),
                     "configuration": (Config, "get_config")}
    
    def __init__(self, num_configs, measurement_spacing, num_warmup_updates,
                 update_method="heatbath", run_parallel=True, rand_seed=-1,
                 verbosity=1):
        """Creates and returns a simulation object
        
        :param num_configs: The number of configurations on which to perform measurements
        :type num_configs: :class:`int`
        :param measurement_spacing: The number of updates between measurements
        :type measurement_spacing: :class:`int`
        :param num_warmup_updates: The number of updates used to thermalize the lattice
        :type num_warmup_updates: :class:`int`
        :param update_method: The method used to update the lattice; current supported methods are "heatbath", "staple_metropolis" and "metropolis"
        :type update_method: :class:`str`
        :param run_parallel: Determines whether OpenMP is used when updating the lattice
        :type run_parallel: :class:`bool`
        :param rand_seed: The random number seed used for performing updates; -1 results in the current time being used
        :type rand_seed: :class:`int`
        :param verbosity: The level of verbosity when peforming the simulation, with 0 producing no output, 1 producing some output and 2 producing the most output, such as details of propagator inversions
        :type verbosity: :class:`int`
        """
        
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
        
        :param L: The lattice spatial extent
        :type L: :class:`int`
        :param T: The lattice temporal extent
        :type T: :class:`int`
        :param action: The gauge action to use in gauge field updates
        :type action: :class:`str`
        :param beta: The inverse coupling to use in the gauge action
        :type beta: :class:`float`
        :param u0: The mean link to use in tadpole improvement
        :type u0: :class:`float`
        :param block_size: The sub-lattice size to use when performing gauge field updates in parallel
        """
        
        self.lattice = Lattice(L, T, beta, u0, action, self.measurement_spacing,
                               self.update_method, self.run_parallel,
                               block_size, self.rand_seed)
    
    def load_ensemble(self, filename):
        """Loads a gauge configuration dataset to use in the simulation
        
        :param filename: The ensemble file
        :type filename: :class:`str`
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
            
    def add_measurement(self, meas_function, meas_type, meas_file, meas_message,
                        kwargs={}):
        """Adds a measurement to the simulation to be performed when the
        simulation is run
        
        :param meas_function: The function defining the measurement
        :type meas_function: :class:`function`
        :param meas_type: The type corresponding the output from the measurement function
        :type meas_type: :class:`type`
        :param meas_file: The :class:`DataSet` file in which to store the measurements
        :type meas_file: :class:`str`
        :param meas_message: The message to display when performing the measurement
        :type meas_message: :class:`str`
        :param kwargs: The keyword arguments for the measurement function.
        :type kwargs: :class:`dict`
        """
        
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
        
        :param timing_run: Performs a number of trial updates and measurements to estimate the total wall clock time of the simulation
        :type timing_run: :class:`bool`
        :param num_timing_configs: The number of updates and measurements used to estimate the total wall clock time
        :type num_timing_configs: :class:`int`
        """
        
        if store_plaquette:
            self.plaquettes = np.zeros(self.num_configs)
        
        t0 = time.time()
        
        if self.verbosity > 0:
            print(self)
        
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
        
        for measurement in self.measurements:
            heading_underline \
              = (len(measurement[1].datatype.__name__) + 21) * "-"
            meas_settings = \
              "{} Measurement Settings\n" \
              "{}\n".format(measurement[1].datatype.__name__, heading_underline)
            
            meas_settings \
              = "".join([meas_settings,
                         "Filename: {}\n".format(measurement[1].filename)])
                
            for key, value in measurement[0].items():
                meas_settings = "".join([meas_settings,
                                        "{}: {}\n".format(key, value)])
                
        
            out = "".join([out, meas_settings])
            
        return out
