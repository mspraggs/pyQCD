from kernel import lattice
import wilslps, propagator, config
import dicts

class Lattice(lattice.Lattice):
    
    def __init__(self, L = 4, T = 8, beta = 5.5, u0 = 1.0, action = "wilson",
                 n_cor = 10, update_method = "heatbath", parallel_updates = True,
                 block_size = 4, rand_seed = -1):
        
        self.beta = beta
        self.u0 = u0
        self.action = action
        self.update_method = update_method
        self.parallel_updates = parallel_updates
        self.block_size = block_size
        self.rand_seed = rand_seed
        
        lattice.Lattice.__init__(self, L, T, beta, u0,
                                 dicts.gauge_actions[action], n_cor,
                                 dicts.update_methods[update_method],
                                 dicts.truefalse[parallel_updates], block_size,
                                 rand_seed)
    
    def get_config(self):
        out = config.Config()
        
        return out
    
    def set_config(self, configuration):
        pass
    
    def save_config(self, filename):
        
        configuration = self.get_config()
        configuration.save(filename)
        
    def load_config(self, filename):
        configuration = config.Config(filename)
        self.set_config(configuration)
    
    def get_wilson_loops(self):
        pass
    
    def get_propagator(self):
        pass
    
    def __str__(self):
        pass
