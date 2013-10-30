import numpy as np

class Config:
    
    def __init__(self, links, L, T, beta, u0, action, n_cor, update_method,
                 parallel_updates, block_size, rand_seed):
        """Create a configuration container."""
        
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
        self.n_cor = n_cor
        self.update_method = update_method
        self.parallel_updates = parallel_updates
        self.block_size = block_size
        self.rand_seed = rand_seed
        
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
        "Mean link (u0): {}\n" \
        "Update method: {}\n" \
        "Measurement spacing: {}\n" \
        "Parallel updates: {}\n" \
        "Parallel update block size: {}\n" \
        "Random number seed: {}".format(self.L, self.T, self.action, self.beta,
                                        self.u0, self.update_method, self.n_cor,
                                        self.parallel_updates, self.block_size,
                                        self.rand_seed)
        
        return out
    
    def av_plaquette(self):
        pass

