from lattice import Lattice
from config import Config
from propagator import Propagator
from twopoint import TwoPoint
from dataset import DataSet
import numpy as np

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
        """
        
        self.num_configs = num_configs
        self.measurement_spacing = measurement_spacing
        self.num_warmup_updates = num_warmup_updates
        
        self.update_method = update_method
        self.run_parallel = run_parallel
        self.rand_seed = rand_seed
        
        self.use_ensemble = False
    
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
                               block_size, rand_seed)
    
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
    
    def add_measurement(self, meas_type, **kwargs):
        pass
    
    def do_measurements(self):
        pass
    
    def run(self, timing_run=False, num_timing_configs=10):
        pass
    
    @classmethod
    def load(cls, xmlfile):
        pass

    def __str__(self):
        pass
