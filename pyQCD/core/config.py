import numpy as np

class Config:
    
        """Create a configuration container."""
    def __init__(self, links, L, T, beta, u0, action):
        
        expected_shape = (T, L, L, L, 4, 3, 3)
        if links.shape != expected_shape:
            raise ValueError("Shape of specified links array, {}, does not "
                             "match the specified lattice extents, {}"
                             .format(links.shape, expected_shape))
        
        self.L = L
        self.T = T
        self.beta = beta
        self.u0 = u0
        self.action = action
        
        self.data = links
    
    def save(self, filename):
        """Documentation"""
        keys = ['L', 'T', 'beta', 'u0', 'action', 'n_cor', 'update_method',
                'parallel_updates', 'block_size', 'rand_seed']
        
        items = [getattr(self, key) for key in keys]
        
        np.savez(filename, header = dict(zip(keys, items)), data = self.data)
        
    @classmethod
    def load(self, filename):
        pass
        
    def save_raw(self, filename):
        np.save(filename, self.data)
    
    def __repr__(self):
        
        out = \
          "Field Configuration Object\n" \
        "--------------\n" \
        "Spatial extent: {}\n" \
        "Temportal extent: {}\n" \
        "Gauge action: {}\n" \
        "Inverse coupling (beta): {}\n" \
        "Mean link (u0): {}".format(self.L, self.T, self.action, self.beta,
                                    self.u0)
        
        return out
    
    def av_plaquette(self):
        pass

