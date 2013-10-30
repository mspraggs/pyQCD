from kernel import lattice
import wilslps, propagator, config
import dicts
import itertools
import numpy as np

class Lattice(lattice.Lattice):
    
    def __init__(self, L=4, T=8, beta=5.5, u0=1.0, action="wilson",
                 n_cor=10, update_method="heatbath", parallel_updates=True,
                 block_size=4, rand_seed=-1):
        """Create a Lattice object.
                 
        :param L: The spatial extent of the lattice
        :type L: :class:`int`
        :param T: The temporal extent of the lattice
        :type T: :class:`int`
        :param beta: The inverse coupling
        :type beta: :class:`float`
        :param u0: The mean link
        :type u0: :class:`float`
        :param action: The gauge action
        :type action: :class:`str`, one of wilson, rectangle_improved or
        twisted_rectangle_improved
        :param n_cor: The number of field configurations between measurements
        :type n_cor: :class:`int`
        :param update_method: The method used to update the field configuration
        :type update_method: :class:`str`, one of heatbath, metropolis or
        staple metropolis
        :param parallel_updates: Specify whether to use parallel updates
        :type parallel_updates: :class:`bool`
        :param block_size: The block edge length used when dividing the lattice
        into domains during parallel updates
        :type block_size: :class:`int`, must be a factor of both :samp:`L` and
        :samp:`T`
        :param rand_seed: A seed for the random number generator
        :type rand_seed: :class:`int`, -1 specifies a random seed, whilst 0 and
        above specifies a pre-determined seed
        """
        
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
        
        raw_links = []
        r = xrange(self.L)
        t = xrange(self.T)
        sites = itertools.product(t, r, r, r, range(4))
        
        links = np.zeros((self.T, self.L, self.L, self.L, 4, 3, 3),
                         dtype=complex)
        
        for t, x, y, z, mu in sites:
            links[t][x][y][z][mu] = np.array(self.get_link([t, x, y, z, mu]))
            
        out = config.Config(links, self.L, self.T, self.beta, self.u0,
                            self.action)
        
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
    
    def __repr__(self):
        
        out = \
          "Lattice Object\n" \
        "--------------\n" \
        "Spatial extent: {}\n" \
        "Temportal extent: {}\n" \
        "Number of sites: {}\n" \
        "Number of links: {}\n" \
        "Gauge action: {}\n" \
        "Inverse coupling (beta): {}\n" \
        "Mean link (u0): {}\n" \
        "Update method: {}\n" \
        "Measurement spacing: {}\n" \
        "Parallel updates: {}\n" \
        "Parallel update block size: {}\n" \
        "Random number seed: {}".format(self.L, self.T, self.L**3 * self.T,
                                        self.L**3 * self.T * 4, self.action,
                                        self.beta, self.u0, self.update_method,
                                        self.n_cor, self.parallel_updates,
                                        self.block_size, self.rand_seed)
        
        return out
