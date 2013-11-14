from lattice import Lattice
from config import Config
from propagator import Propagator
from wilslps import WilsonLoops
from dataset import DataSet
import numpy as np
import sys
import time
import inspect

class Simulation(object):
    
    def __init__(self, num_configs, measurement_spacing, num_warmup_updates,
                 update_method="heatbath", run_parallel=True, rand_seed=-1,
                 verbosity=1):
        """Creates and returns a simulation object
        
        :param num_configs: The number of configurations on which to perform
        measurements
        :type num_configs: :class:`int`
        :param measurement_spacing: The number of updates between measurements
        :type measurement_spacing: :class:`int`
        :param num_warmup_updates: The number of updates used to thermalize
        the lattice
        :type num_warmup_updates: :class:`int`
        :param update_method: The method used to update the lattice; current
        supported methods are "heatbath", "staple_metropolis" and "metropolis"
        :type update_method: :class:`str`
        :param run_parallel: Determines whether OpenMP is used when updating
        the lattice
        :type run_parallel: :class:`bool`
        :param rand_seed: The random number seed used for performing updates;
        -1 results in the current time being used
        :type rand_seed: :class:`int`
        :param verbosity: The level of verbosity when peforming the simulation,
        with 0 producing no output, 1 producing some output and 2 producing the
        most output, such as details of propagator inversions
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
        
        self.measurements = {}
    
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
        :param block_size: The sub-lattice size to use when performing gauge
        field updates in parallel
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
            AttributeError("A lattice must be defined before an ensemble may "
                           "be loaded.")
        
        ensemble = DataSet.load(Config, filename)
        
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
    
    def add_measurement(self, meas_type, meas_file, **kwargs):
        """Adds a measurement to the simulation to be performed when the
        simulation is run
        
        Possible parameters (depending on :samp:`meas_type`):
        
        :param meas_type: The class corresponding to the measurement to be
        performed
        :type meas_type: :class:`type`
        :param meas_file: The :class:`DataSet` file in which to store the
        measurement
        :param mass: The mass to use if a propagator is to be calculated
        :type mass: :class:`float`
        :param source_site: The source site to use when computing a propagator
        (default it [0, 0, 0, 0])
        :type source_site: :class:`list`
        :param num_field_smears: The number of times to stout smear the gauge
        (default is 0)
        field before performing a measurement
        :type num_field_smears: :class:`int`
        :param field_smearing_param: The stout smearing parameter default is 1.0
        :type field_smearing_param: :class:`float`
        :param num_source_smears: The number of Jacobi smears to apply to the
        source when computing a propagator (default is 0)
        :type num_source_smears: :class:`int`
        :param source_smearing_param: The smearing parameter to use when
        smearing the source (default is 1.0)
        :type source_smearing_param: :class:`float`
        :param num_sink_smears: The number of Jacobi smears to apply to the
        sink when computing a propagator (default is 0)
        :type num_sink_smears: :class:`int`
        :param sink_smearing_param: The smearing parameter to use when
        smearing the sink (default is 1.0)
        :type sink_smearing_param: :class:`float`
        :param solver_method: The method to use when computing a propagator, may
        either be "bicgstab" or "conjugate_gradient" (default "bicgstab")
        :type sink_smearing_param: :class:`str`
        """
        
        if meas_type == Config:
            dataset = DataSet(meas_type, meas_file)
            message = "Saving field configuration"
            function = "get_config"
            
            self.measurements.update([(message, (kwargs, dataset, function))])
        
        elif meas_type == WilsonLoops:
            dataset = DataSet(meas_type, meas_file)
            message = "Computing wilson loops"
            function = "get_wilson_loops"
            
            self.measurements.update([(message, (kwargs, dataset, function))])
            
        elif meas_type == Propagator:
            dataset = DataSet(meas_type, meas_file)
            message = "Computing propagator"
            function = "get_propagator"
            
            if self.verbosity > 0:
                kwargs.update([("verbosity", self.verbosity - 1)])
            else:
                kwargs.update([("verbosity", 0)])
            
            self.measurements.update([(message, (kwargs, dataset, function))])
            
        else:
            raise TypeError("Measurement data type {} is not understood"
                            .format(meas_type))
    
    def _do_measurements(self):
        """Iterate through self.measurements and gather results"""
        
        keys = self.measurements.keys()
        
        for key in keys:
            print("%s...".format(key))
            
            measurement = getattr(self.lattice, self.measurements[key][2]) \
              (self.measurements[key][0])
            
            self.measurements[key][1].add_datum(measurement)
              
            print("Done!")
    
    def run(self, timing_run=False, num_timing_configs=10):
        """Runs the simulation
        
        :param timing_run: Performs a number of trial updates and measurements
        to estimate the total wall clock time of the simulation
        :type timing_run: :class:`bool`
        :param num_timing_configs: The number of updates and measurements used
        to estimate the total wall clock time
        :type num_timing_configs: :class:`int`
        """
        
        pass
    
    @classmethod
    def load(cls, xmlfile):
        pass

    def __str__(self):
        
        out = \
          """Simulation Settings
        -------------------
        Number of configurations: {}
        Measurement spacing: {}
        Thermalization updates: {}
        Update method: {}
        """
