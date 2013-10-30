from kernel import lattice
import wilslp, propagator, config, header

class Lattice(lattice.Lattice):
    
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
