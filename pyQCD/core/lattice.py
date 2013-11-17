from kernel import lattice
import wilslps, propagator, config
import dicts
import itertools
import numpy as np

class Lattice(lattice.Lattice):
    """docstring"""
    
    def __init__(self, L=4, T=8, beta=5.5, u0=1.0, action="wilson",
                 n_cor=10, update_method="heatbath", parallel_updates=True,
                 block_size=None, rand_seed=-1):
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
        :type action: :class:`str`, one of wilson, rectangle_improved or twisted_rectangle_improved
        :param n_cor: The number of field configurations between measurements
        :type n_cor: :class:`int`
        :param update_method: The method used to update the field configuration
        :type update_method: :class:`str`, one of heatbath, metropolis or staple metropolis
        :param parallel_updates: Specify whether to use parallel updates
        :type parallel_updates: :class:`bool`
        :param block_size: The block edge length used when dividing the lattice into domains during parallel updates
        :type block_size: :class:`int`, must be a factor of both :samp:`L` and :samp:`T`
        :param rand_seed: A seed for the random number generator
        :type rand_seed: :class:`int`, -1 specifies a random seed, whilst 0 and above specifies a pre-determined seed
        :returns: :class:`Lattice`
        :raises: ValueError
        """
        
        if block_size == None:
            block_size = 3
            
            while L % block_size > 0 or T % block_size > 0:
                block_size += 1
                if block_size > L or block_size > T:
                    raise ValueError("Lattice shape {} cannot accomodate "
                                     "sub-lattices with side-length {}"
                                     .format((T, L, L, L), block_size))
        
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
        
    def get_link(self, link):
        """Returns the specified gauge field link
        
        :param link: The site and dimension of the link to be returned
        :type link: :class:`list` of five ints, of the form [t, x, y, z, mu]
        :returns: :class:`numpy.ndarray`
        """
        
        return np.array(lattice.Lattice.get_link(self, link))
    
    def set_link(self, link, matrix):
        
        lattice.Lattice.set_link(self, link, matrix.tolist())
    
    def get_config(self):
        """Returns the current field configuration.
        
        :returns: :class:`Config`
        """
        r = xrange(self.L)
        t = xrange(self.T)
        sites = itertools.product(t, r, r, r, range(4))
        
        links = np.zeros((self.T, self.L, self.L, self.L, 4, 3, 3),
                         dtype=complex)
        
        for t, x, y, z, mu in sites:
            links[t][x][y][z][mu] \
              = np.array(lattice.Lattice.get_link(self, [t, x, y, z, mu]))
            
        out = config.Config(links, self.L, self.T, self.beta, self.u0,
                            self.action)
        
        return out
    
    def set_config(self, configuration):
        """Sets the current field configuration
        
        :param configuration: The field configuration
        :type configuration: :class:`Config`
        :raises: ValueError
        """
        expected_shape = (self.T, self.L, self.L, self.L, 4, 3, 3)
        
        if configuration.data.shape != expected_shape:
            raise ValueError("Shape of field configuration, {}, does not "
                             "match the current lattice shape, {}"
                             .format(configuration.data.shape, expected_shape))
        
        r = xrange(self.L)
        t = xrange(self.T)
        sites = itertools.product(t, r, r, r, range(4))
        
        links = np.zeros((self.T, self.L, self.L, self.L, 4, 3, 3),
                         dtype=complex)
        
        for t, x, y, z, mu in sites:
            link_matrix = configuration.data[t][x][y][z][mu].tolist()
            lattice.Lattice.set_link(self, [t, x, y, z, mu], link_matrix)
    
    def save_config(self, filename):
        """Saves the current field configuration to a file
        
        :param filename: The file to which the config will be saved
        :type filename: :class:`str`
        """
        
        configuration = self.get_config()
        configuration.save(filename)
        
    def load_config(self, filename):
        """Loads the configuration in the specified file
        
        :param filename: The file from which the config will be loaded
        :type filename: :class:`str`
        """
        
        configuration = config.Config.load(filename)
        self.set_config(configuration)
        
    def update(self):
        """Update the gauge field using the method specified in the
        Lattice constructor        
        """
        
        lattice.Lattice.update(self)
        
    def next_config(self):
        """Updates the gauge field by a number of times specified
        by measurement spacing in the Lattice constructor
        """
        
        lattice.Lattice.next_config(self)
        
    def thermalize(self, num_updates):
        """Thermalizes the lattice by ensuring num_updates have been
        performed
        
        :param num_updates: The number of updates to ensure have been performed
        :type num_updates: :class:`int`
        """
        
        lattice.Lattice.thermalize(self, num_updates)
        
    def get_plaquette(self, site, dim1, dim2):
        """Computes and returns the value of the specified plaquette
        
        :param site: The site specifying the corner of the plaquette
        :type site: :class:`list` of the form [t, x, y, z]
        :param dim1: The first dimension specifying the plane in which the plaquette lies
        :type dim1: :class:`int`
        :param dim2: The second dimension specifying the plane in which the plaquette lies
        :type dim2: :class:`int`
        :returns: :class:`float`
        """
        
        return lattice.Lattice.get_plaquette(self, site, dim1, dim2)
        
    def get_rectangle(self, site, dim1, dim2):
        """Computes and returns the value of the specified 2 x 1 rectangle
        
        :param site: The site specifying the corner of the rectangle
        :type site: :class:`list` of the form [t, x, y, z]
        :param dim1: The first dimension specifying the longer rectangle edge
        :type dim1: :class:`int`
        :param dim2: The second dimension specifying the shorter edge of the rectangle
        :type dim2: :class:`int`
        :returns: :class:`float`
        """
        
        return lattice.Lattice.get_rectangle(self, site, dim1, dim2)
        
    def get_twist_rect(self, site, dim1, dim2):
        """Computes and returns the value of the specified 2 x 1 twisted
        rectangle
        
        :param site: The site specifying the corner of the twisted rectangle
        :type site: :class:`list` of the form [t, x, y, z]
        :param dim1: The first dimension specifying the longer twisted rectangle edge
        :type dim1: :class:`int`
        :param dim2: The second dimension specifying the shorter edge of the twisted rectangle
        :type dim2: :class:`int`
        :returns: :class:`float`
        """
        
        return lattice.Lattice.get_twist_rect(self, site, dim1, dim2)
        
    def get_wilson_loop(self, corner, r, t, dim, num_smears=0,
                        smearing_param=1.0):
        
        """Computes and returns the value of the specified Wilson loop
        
        :param corner: The starting corner of the Wilson loop
        :type corner: :class:`list` of the form [t, x, y, z]
        :param r: The size of the loop in the spatial direction
        :type r: :class:`int`
        :param t: The size of the loop in the temporal direction
        :type t: :class:`int`
        :param dim: The spatial dimension of the Wilson loop
        :type dim: :class:`int`
        :param num_smears: The number of stout gauge field smears to perform before computing the Wilson loop
        :type num_smears: :class:`int`
        :param smearing_param: The stout gauge field smearing parameter
        :type smearing_param: :class:`float`
        :returns: :class:`float`
        """
        
        return lattice.Lattice.get_wilson_loop(self, corner, r, t, dim,
                                               num_smears, smearing_param)
        
    def get_av_plaquette(self):
        """Computes the plaquette expectation value
        
        :returns: :class:`float`
        """
        
        return lattice.Lattice.get_av_plaquette(self)
    
    def get_av_rectangle(self):
        """Computes the rectangle expectation value
        
        :returns: :class:`float`
        """
        
        return lattice.Lattice.get_av_rectangle(self)
    
    def get_av_wilson_loop(self, r, t, num_smears=0, smearing_param=1.0):
        """Computes the average wilson loop of a given size
        
        :param r: The spatial extent of the Wilson loop
        :type r: :class:`int`
        :param t: The temporal extent of the Wilson loop
        :type t: :class:`int`
        :param num_smears: The number of stout gauge field smears to perform before computing the Wilson loops
        :type num_smears: :class:`int`
        :param smearing_param: The stout gauge field smearing parameter
        :type smearing_param: :class:`float`
        :returns: :class:`float`
        """
        
        return lattice.Lattice.get_av_wilson_loop(self, r, t, num_smears,
                                                  smearing_param)
    
    def get_wilson_loops(self, num_field_smears=0, field_smearing_param=1.0):
        """Calculates and returns all Wilson loops of size m x n,
        with m = 0, 1, ... , L and n = 0, 1, ... , T.
        
        :param num_field smears: The number of stout smears to perform
        :type num_field_smears: :class:`int`
        :param field_smearing_param: The stout smearing parameter to use
        :type field_smearing_param: :class:`float`
        :returns: :class:`WilsonLoops`
        """
        
        loops = np.zeros((self.L, self.T))
        
        for r in xrange(self.L):
            for t in xrange(self.T):
                loops[r, t] \
                  = lattice.Lattice.get_av_wilson_loop(self, r, t,
                                                       num_field_smears,
                                                       field_smearing_param)
                
        out = wilslps.WilsonLoops(loops, self.L, self.T, self.beta, self.u0,
                                  self.action, num_field_smears,
                                  field_smearing_param)
        
        return out
    
    def get_propagator(self, mass,
                       source_site = [0, 0, 0, 0],
                       num_field_smears = 0,
                       field_smearing_param = 1.0,
                       num_source_smears = 0,
                       source_smearing_param = 1.0,
                       num_sink_smears = 0,
                       sink_smearing_param = 1.0,
                       solver_method = "bicgstab",
                       verbosity = 0):
        """Create a field configuration container.
                 
        :param mass: The bare quark mass
        :type mass: :class:`float`
        :param source_site: The source site used when doing the inversion
        :type source_site: :class:`list`
        :param num_field_smears: The number of stout smears applied when computing the propagator
        :type num_field_smears: :class:`int`
        :param field_smearing_param: The stout smearing parameter
        :type field_smearing_param: :class:`float`
        :param num_source_smears: The number of Jacobian smears performed on the source when computing the propagator
        :type num_source_smears: :class:`int`
        :param source_smearing_param: The Jacobi smearing parameter used when smearing the source
        :type source_smearing_param: :class:`float`
        :param num_sink_smears: The number of Jacobian smears performed on the source when computing the propagator
        :type num_sink_smears: :class:`int`
        :param sink_smearing_param: The Jacobi smearing parameter used when smearing the source
        :type sink_smearing_param: :class:`float`
        :param solver_method: The method to use when performing the inversion
        :type solver_method: :class:`str`
        :param verbosity: Determines the degree of output printed to the screen
        :type verbosity: :class:`int`
        :returns: :class:`Propagator`
        """
        
        raw_propagator \
          = np.array(lattice.Lattice \
                     .get_propagator(self,
                                     mass,
                                     1.0,
                                     source_site,
                                     num_field_smears,
                                     field_smearing_param,
                                     num_source_smears,
                                     source_smearing_param,
                                     num_sink_smears,
                                     sink_smearing_param,
                                     dicts.solver_methods[solver_method],
                                     verbosity))
        
        prop = np.swapaxes(np.reshape(raw_propagator, (self.T, self.L, self.L,
                                                       self.L, 4, 3, 4, 3)), 5,
                                                       6)
        
        out = propagator.Propagator(prop, self.L, self.T, self.beta, self.u0,
                                    self.action, mass, source_site,
                                    num_field_smears, field_smearing_param,
                                    num_source_smears, source_smearing_param,
                                    num_sink_smears, sink_smearing_param)
        
        return out
    
    def get_av_link(self):
        """Computes the mean of the real part of the trace of each link matrix
        
        :returns: :class:`float`
        """
        
        return lattice.Lattice.get_av_link(self)
    
    def __str__(self):
        
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
