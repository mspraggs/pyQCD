import itertools
import numpy as np

from kernel import lattice
import wilslps
import propagator
import config
import dicts

class Lattice(lattice.Lattice):
    """Encapsulates a hypercubic lattice and associated SU(3) gauge field
    
    This class supplies functions to create, thermalize and update an SU(3)
    gauge field configuration, as well as perform various measurements
    upon said configuration, such as the computation of Wilson loops and
    propagators. This is in contrast to the Config class, which simply acts
    to encapsulate the raw SU(3) data. The bulk of the code for this class
    is written in C++ and wrapped in Boost::Python.
    
    Attributes:
      L (int): The spatial extent of the lattice
      T (int): The temporal extent of the lattice
      beta (float): The inverse coupling of the gauge action
      ut (float): The spatial mean link/tadpole improvement factor.
      us (float): The temporal mean link/tadpole improvement factor.
      chi (float): The anisotropy factor, equal to the spatial lattice spacing
        divided by the temporal lattice spacing.
      action (str): The gauge action to use when updating the lattice
        (see note 1).
      n_cor (int): The number of configurations between measurements
      update_method (str): The algorithm to use when updating the
        gauge field configuration (see note 2).
      parallel_updates (bool): Determines whether to partition the
        lattice into a red-black/checkerboard pattern, then perform
        updates on the red blocks in parallel before updating the
        black blocks. (Requires OpenMP to work.)
      block_size (int): The side-length of the blocks used to partition
        the lattice.
      rand_seed (int): The seed to be used by the random number
        generator.     
    
    Args:
      L (int, optional): The spatial extent of the lattice
      T (int, optional): The temporal extent of the lattice
      beta (float, optional): The inverse coupling of the gauge action
      ut (float, optional): The spatial mean link/tadpole improvement factor.
      us (float, optional): The temporal mean link/tadpole improvement factor.
      chi (float): The anisotropy factor, equal to the spatial lattice spacing
        divided by the temporal lattice spacing.
      action (str, optional): The gauge action to use when updating the lattice.
        Currently "wilson", "rectangle_improved" and
        "twisted_rectangle_improved" are supported (see note 1).
      n_cor (int, optional): The number of configurations between measurements
      update_method (str, optional): The algorithm to use when updating the
        gauge field configuration. Currently "heatbath", "metropolis"
        and "staple_metropolis" are supported (see note 2).
      parallel_updates (bool, optional): Determines whether to partition the
        lattice into a red-black/checkerboard pattern, then perform
        updates on the red blocks in parallel before updating the
        black blocks. (Requires OpenMP to work.)
      block_size (int, optional): The side-length of the blocks to partition the
        lattice into when performing parallel updates.
      rand_seed (int, optional): The seed to be used by the random number
        generator (-1 specifies that a seed based on the time should
        be used).
         
    Returns:
      Lattice: The created lattice object
      
    Raises:
      ValueError: Lattice shape cannot accomodate sub-lattices with
        the specified side-length
        
    Examples:
      Create an 8^3 x 16 lattice with the Symanzik rectange-improved
      action, an inverse coupling 4.26 and a tadpole-improvement factor
      of 0.852.
      
      >>> import pyQCD
      >>> lattice = pyQCD.Lattice(L=8, T=16, action="rectangle_improved"
      ...           beta=4.26, ut=0.852, us=0.852)
        
    Notes:
      1. The "wilson" action is the well-known Wilson gauge action. The
         action specified by "rectangle_improved" is the Symanzik
         rectangle-improved gauge action. This is an a^2 improved gauge
         action. The action specified by "twisted_rectangle_improved"
         uses the twisted rectangle operator to eliminate
         discretisation errors of order a^2. Staple computation for this
         operator is difficult, and as such the heatbath algorithm is
         not capable of using this action. See Lepage, "Lattice QCD
         for novices", for more details on these actions.
      2. The update method "metropolis" computes the local change in the
         action without using the staples for the given link. This is
         highly inefficient, but inescapable when using actions such as
         twisted rectangle imporoved action where the staples are
         difficult to define and compute. The "staple_metropolis"
         algorithm computes the staples for each link, thus reducing the
         number of computations and improving efficiency. The "heatbath"
         method is to be preferred where the computation of staples is
         possible, being the most efficient algorithm.
    """
    
    def __init__(self, L=4, T=8, beta=5.5, ut=1.0, us=1.0, chi=1.0,
                 action="wilson", n_cor=10, update_method="heatbath",
                 parallel_updates=True, block_size=None, rand_seed=-1):
        """Constructor for pyQCD.Lattice (see help(pyQCD.Lattice))"""
        if block_size == None:
            block_size = 3
            
            while L % block_size > 0 or T % block_size > 0:
                block_size += 1
                if block_size > L or block_size > T:
                    raise ValueError("Lattice shape {} cannot accomodate "
                                     "sub-lattices with side-length {}"
                                     .format((T, L, L, L), block_size))
        
        self.beta = beta
        self.ut = ut
        self.us = us
        self.chi = chi
        self.action = action
        self.update_method = update_method
        self.parallel_updates = parallel_updates
        self.block_size = block_size
        self.rand_seed = rand_seed
        self.shape = (T, L, L, L)
        
        lattice.Lattice.__init__(self, L, T, beta, ut, us, chi,
                                 dicts.gauge_actions[action], n_cor,
                                 dicts.update_methods[update_method],
                                 dicts.truefalse[parallel_updates], block_size,
                                 rand_seed)
        
    def get_link(self, link):
        """Returns the specified gauge field link
        
        Args:
          link (list): The link to return. The supplied list should be of
            the form [t, x, y, z, mu].
          
        Returns:
          numpy.ndarray: The SU(3) gauge field link.
          
          The returned link will be a 3 x 3 numpy array
          
        Examples:
           Create a basic lattice and return the link at the origin
           in the time direction.
           
           >>> import pyQCD
           >>> lattice = pyQCD.Lattice()
           >>> lattice.get_link([0, 0, 0, 0, 0])
           array([[ 1.+0.j,  0.+0.j,  0.+0.j],
                  [ 0.+0.j,  1.+0.j,  0.+0.j],
                  [ 0.+0.j,  0.+0.j,  1.+0.j]])
        """
        
        return np.array(lattice.Lattice.get_link(self, link))
    
    def set_link(self, link, matrix):
        """Sets the specified link to the value specified in matrix
        
        Args:
          link (list): The link to set. The supplied list should be of
            the form [t, x, y, z, mu].
          matrix (numpy.ndarray): 3 x 3 numpy array specifying the link
            matrix.
            
        Examples:
          Set the temporal link at the origin to the identity for
          the created lattice.
          
          >>> import pyQCD
          >>> import numpy
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> lattice.set_link([0, 0, 0, 0, 0], numpy.identity(3))
        """
        
        lattice.Lattice.set_link(self, link, matrix.tolist())
    
    def get_config(self):
        """Returns the current field configuration.
        
        Returns:
          Config: The current gauge field configuration
          
        Examples:
          Create a lattice, thermalize it, then retrieve the current
          gauge field configuration.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> lattice.get_config()
          Field Configuration Object
          --------------------------
          Spatial extent: 4
          Temportal extent: 8
          Gauge action: wilson
          Inverse coupling (beta): 5.5
          Mean temporal link (ut): 1.0
          Mean spatial link (us): 1.0
          Anisotropy factor (chi): 1.0
        """
        r = xrange(self.L)
        t = xrange(self.T)
        sites = itertools.product(t, r, r, r, range(4))
        
        links = np.zeros((self.T, self.L, self.L, self.L, 4, 3, 3),
                         dtype=complex)
        
        for t, x, y, z, mu in sites:
            links[t][x][y][z][mu] \
              = np.array(lattice.Lattice.get_link(self, [t, x, y, z, mu]))
            
        out = config.Config(links, self.L, self.T, self.beta, self.ut,
                            self.us, self.chi, self.action)
        
        return out
    
    def set_config(self, configuration):
        """Sets the current field configuration
        
        Args:
          configuration (Config): The pyQCD.Config object containing
            the gauge field configuration to use.
            
        Examples:
          Load a gauge field configuration from disk and load it into
          the current lattice object.
          
          >>> import pyQCD
          >>> config = pyQCD.Config.load("myconfig.npz")
          >>> lattice = pyQCD.Lattice()
          >>> lattice.set_config(config)
        """
        
        r = xrange(self.L)
        t = xrange(self.T)
        sites = itertools.product(t, r, r, r, range(4))
        
        links = np.zeros((self.T, self.L, self.L, self.L, 4, 3, 3),
                         dtype=complex)
        
        for t, x, y, z, mu in sites:
            link_matrix = configuration.data[t][x][y][z][mu].tolist()
            lattice.Lattice.set_link(self, [t, x, y, z, mu], link_matrix)
    
    def save_config(self, filename):
        """Saves the current field configuration to a numpy zip file
        
        Args:
          filename (str): The file to save the configuration to.
          
        Examples:
          Create a lattice, thermalize and save the resulting config
          to myconfig.npz.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> lattice.save_config("myconfig")
        """
        
        configuration = self.get_config()
        configuration.save(filename)
        
    def load_config(self, filename):
        """Loads the configuration in the specified file
        
        Args:
          filename (str): The filename of the field configuration
            (as a numpy zipped archive).
            
        Examples:
          Create a lattice and load the configuration in myconfig.npz
          from disk into it.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.load_config("myconfig.npz")
        """
        
        configuration = config.Config.load(filename)
        self.set_config(configuration)
        
    def update(self):
        """Update the gauge field using the method specified in the
        Lattice constructor
        
        Examples:
          Create a lattice and update it 10 times. After each update,
          print the average plaquette.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice(L=8, T=8)
          >>> for i in xrange(10):
          ...     lattice.update()
          ...     print(lattice.get_av_plaquette())
        """
        
        lattice.Lattice.update(self)
        
    def next_config(self):
        """Updates the gauge field by a number of times specified
        by measurement spacing (n_cor) in the Lattice constructor
        
        Examples:
          Create a lattice and generate 10 configurations, each
          separated by 10 field updates. After each configuration
          is generated, print the average plaquette.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice(L=8, T=8)
          >>> for i in xrange(10):
          ...     lattice.next_config()
          ...     print(lattice.get_av_plaquette())
        """
        
        lattice.Lattice.next_config(self)
        
    def thermalize(self, num_updates):
        """Thermalizes the lattice by ensuring num_updates have been
        performed
        
        Args:
          num_updates (int): The number of updates to ensure have been
            performed.
            
        Examples:
           Create a lattice and thermalize it with 100 updates, then print
           the average plaquette.
           
           >>> import pyQCD
           >>> lattice = pyQCD.Lattice()
           >>> lattice.thermalize(100)
           >>> print(lattice.get_av_plaquette())
        """
        
        lattice.Lattice.thermalize(self, num_updates)
        
    def get_plaquette(self, site, dim1, dim2):
        """Computes and returns the value of the specified plaquette
        
        Args:
          site (list): The site specifying the corner of the plaquette,
            of the form [t, x, y, z]
          dim1 (int): The first dimension specifying the plane in which the
            plaquette lies
          dim2 (int): The second dimension specifying the plane in which the
            plaquette lies
          
        Returns:
          float: The plaquette values.
          
        Examples:
          Create a lattice, thermalize it, then compute the plaquette sited
          at [0, 0, 0, 0] in the t and x plane (i.e. P_01(0, 0, 0, 0)).
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> print(lattice.get_plaquette([0, 0, 0, 0], 0, 1))
          0.21414804616056343
        """
        
        return lattice.Lattice.get_plaquette(self, site, dim1, dim2)
        
    def get_rectangle(self, site, dim1, dim2):
        """Computes and returns the value of the specified 2 x 1 rectangle
        
        Args:
          site (list): The site specifying the corner of the rectangle,
            of the form [t, x, y, z]
          dim1 (int): The first dimension specifying the plane in which the
            rectangle lies
          dim2 (int): The second dimension specifying the plane in which the
            rectangle lies
          
        Returns:
          float: The rectangle values.
          
        Examples:
          Create a lattice, thermalize it, then compute the rectangle sited
          at [0, 0, 0, 0] in the t and x plane (i.e. R_01(0, 0, 0, 0)).
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> print(lattice.get_rectangle([0, 0, 0, 0], 0, 1))
          0.501128452826521
        """
        
        return lattice.Lattice.get_rectangle(self, site, dim1, dim2)
        
    def get_twist_rect(self, site, dim1, dim2):
        """Computes and returns the value of the specified 2 x 1 twisted
        rectangle
        
        Args:
          site (list): The site specifying the corner of the twisted rectangle,
            of the form [t, x, y, z]
          dim1 (int): The first dimension specifying the plane in which the
            twisted rectangle lies
          dim2 (int): The second dimension specifying the plane in which the
            twisted rectangle lies
          
        Returns:
          float: The twisted rectangle values.
          
        Examples:
          Create a lattice, thermalize it, then compute the twisted rectangle
          sited at [0, 0, 0, 0] in the t and x plane (i.e. T_01(0, 0, 0, 0)).
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> print(lattice.get_twisted rect([0, 0, 0, 0], 0, 1))
          -0.10429666482691154
        """
        
        return lattice.Lattice.get_twist_rect(self, site, dim1, dim2)
        
    def get_wilson_loop(self, corner, r, t, dim, num_smears=0,
                        smearing_param=1.0):
        """Computes and returns the value of the specified Wilson loop
                    
        Args:
          corner (list): The starting corner of the Wilson loop, of the form
            [t, x, y, z].
          r (int): The size of the loop in the spatial direction
          t (int): The size of the loop in the temporal direction
          dim (int): The spatial dimension of the Wilson loop
          num_smears (int, optional): The number of stout gauge field smears to
            perform before computing the Wilson loop
          smearing_param (float, optional): The stout gauge field smearing
            parameter
          
        Returns:
          float: The value of the Wilson loop
          
        Examples:
          Create a lattice, thermalize it, then compute the wilson loop sited
          at [4, 0, 2, 0] in the t and x plane (i.e. W_01(4, 0, 2, 0)), with
          spatial width 2 and temporal width 4.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> lattice.get_wilson_loop([4, 0, 2, 0], 2, 4, 1)
          -0.19872418745939716
        """
        
        return lattice.Lattice.get_wilson_loop(self, corner, r, t, dim,
                                               num_smears, smearing_param)
        
    def get_av_plaquette(self):
        """Computes the plaquette expectation value
        
        Returns:
          float: The average value of the plaquette
            
        Examples:
          Create a lattice, thermalize and compute the average plaquette
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice(L=8, T=8)
          >>> lattice.thermalize(100)
          >>> lattice.get_av_plaquette()
          0.49751028964943506
        """
        
        return lattice.Lattice.get_av_plaquette(self)
    
    def get_av_rectangle(self):
        """Computes the rectangle expectation value
        
        Returns:
          float: The average value of the rectangle
            
        Examples:
          Create a lattice, thermalize and compute the average rectangle
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice(L=8, T=8)
          >>> lattice.thermalize(100)
          >>> lattice.get_av_rectangle()
          0.2605558875384393
        """
        
        return lattice.Lattice.get_av_rectangle(self)
    
    def get_av_wilson_loop(self, r, t, num_smears=0, smearing_param=1.0):
        """Computes the average wilson loop of a given size
        
        Args:
          r (int): The spatial extent of the Wilson loop
          t (int): The temporal extent of the Wilson loop
          num_smears (int, optional): The number of stout gauge field smears to perform
            before computing the Wilson loops
          smearing_param (float, optional): The stout gauge field smearing parameter
        
        Returns:
          float: The value of the average Wilson loop
          
        Examples:
          Create a lattice, thermalize and compute the average Wilson loop
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> lattice.get_av_wilson_loop(3, 3)
          0.007159100123379076
        """
        
        return lattice.Lattice.get_av_wilson_loop(self, r, t, num_smears,
                                                  smearing_param)
    
    def get_wilson_loops(self, num_field_smears=0, field_smearing_param=1.0):
        """Calculates and returns all Wilson loops of size m x n,
        with m = 0, 1, ... , L and n = 0, 1, ... , T.
        
        Args:
          num_field_smears (int, optional): The number of stout field smears to
            perform on the gauge field prior to computing the Wilson loops
          field_smearing_param (float, optional): The stout field smearing
            parameter
          
        Returns:
          WilsonLoops: The Wilson loops object encapsulating the Wilson loops
          
        Examples:
          Create a lattice object, thermalize it, then extract the Wilson
          loops, smearing the gauge field a bit beforehand
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> wilson_loops = lattice.get_wilson_loops(2, 0.4)
        """
        
        loops = np.zeros((self.L, self.T))
        
        for r in xrange(self.L):
            for t in xrange(self.T):
                loops[r, t] \
                  = lattice.Lattice.get_av_wilson_loop(self, r, t,
                                                       num_field_smears,
                                                       field_smearing_param)
                
        out = wilslps.WilsonLoops(loops, self.L, self.T, self.beta, self.ut,
                                  self.us, self.chi, self.action,
                                  num_field_smears, field_smearing_param)
        
        return out
    
    def get_wilson_propagator(self, mass,
                              source_site=[0, 0, 0, 0],
                              num_field_smears=0,
                              field_smearing_param=1.0,
                              source_smear_type="jacobi",
                              num_source_smears=0,
                              source_smearing_param=1.0,
                              sink_smear_type="jacobi",
                              num_sink_smears=0,
                              sink_smearing_param=1.0,
                              solver_method="conjugate_gradient",
                              boundary_conditions = [-1, 1, 1, 1],
                              precondition=False,
                              max_iterations=1000,
                              tolerance=1e-8,
                              verbosity=0):
        """Compute the Wilson propagator using the Wilson fermion action
                       
        Args:
          mass (float): The bare quark mass
          source_site (list, optional): The source site to use when doing the
            inversion
          num_field_smears (int, optional): The number of stout field smears
            applied before doing the inversion
          field_smearing_param (float, optional): The stout field smearing
            parameter to use before doing the inversion
          source_smear_type (str, optional): The type of smearing to apply
            to the source. Currently "jacobi" is the only supported smearing
            type.
          num_source_smears (int, optional): The number of Jacobi smears to apply
            to the source before inverting.
          source_smearing_param (float, optional): The Jacobi field smearing
            parameter to use before doing the inversion
          sink_smear_type (str, optional): The type of smearing to apply
            to the sink. Currently "jacobi" is the only supported smearing
            type.
          num_sink_smears (int, optional): The number of Jacobi smears to apply
            to the sink before inverting.
          sink_smearing_param (float, optional): The Jacobi field smearing
            parameter to use before doing the inversion
          solver_method (str, optional): The algorithm to use when doing the
            inversion. Currently "conjugate_gradient" and "bicgstab" are
            available.
          boundary_conditions (list, optional): The phase factors to apply to
            the fermion fields at the boundaries of the lattice.
          precondition (bool, optional): Determines whether the preconditioned
            form of the Dirac operator is used.
          max_iterations (int, optional): The maximum number of iterations the
            solver will perform before giving up.
          tolerance (float, optional): The maximum residual the solver will
            allow before considering convergence to have been reached.
          verbosity (int, optional): Determines how much inversion is outputted
            during the inversion. Values greater than one produce output, whilst
            1 or 0 will produce no output.
            
        Returns:
          Propagator: The propagator encapsulated in a Propagator object
          
        Examples:
          Create a lattice, thermalize it, and invert on a smeared source.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> prop = lattice.get_wilson_propagator(0.4, num_source_smears=2,
          ...                                      source_smearing_param=0.4)
        """
        
        if verbosity > 0:
            verbosity -= 1
        
        raw_propagator \
          = np.array(lattice.Lattice \
                     .get_wilson_propagator(
                         self, mass, source_site, num_field_smears,
                         field_smearing_param,
                         dicts.smearing_types[source_smear_type],
                         num_source_smears, source_smearing_param,
                         dicts.smearing_types[sink_smear_type], num_sink_smears,
                         sink_smearing_param, dicts.solver_methods[solver_method],
                         boundary_conditions, dicts.truefalse[precondition],
                         max_iterations, tolerance, verbosity))
        
        prop = np.swapaxes(np.reshape(raw_propagator, (self.T, self.L, self.L,
                                                       self.L, 4, 3, 4, 3)), 5,
                                                       6)
        
        out = propagator.Propagator(prop, self.L, self.T, self.beta, self.ut,
                                    self.us, self.chi, self.action, "wilson",
                                    mass, {}, boundary_conditions, source_site,
                                    num_field_smears, field_smearing_param,
                                    source_smear_type, num_source_smears,
                                    source_smearing_param, sink_smear_type,
                                    num_sink_smears, sink_smearing_param)
        
        return out
    
    def get_hamberwu_propagator(self, mass,
                                source_site=[0, 0, 0, 0],
                                num_field_smears=0,
                                field_smearing_param=1.0,
                                source_smear_type="jacobi",
                                num_source_smears=0,
                                source_smearing_param=1.0,
                                sink_smear_type="jacobi",
                                num_sink_smears=0,
                                sink_smearing_param=1.0,
                                solver_method="conjugate_gradient",
                                boundary_conditions = [-1, 1, 1, 1],
                                precondition=False,
                                max_iterations=1000,
                                tolerance=1e-8,
                                verbosity=0):
        """Compute the Hamber-Wu propagator using the Hamber-Wu fermion action
                       
        Args:
          mass (float): The bare quark mass
          source_site (list, optional): The source site to use when doing the
            inversion
          num_field_smears (int, optional): The number of stout field smears
            applied before doing the inversion
          field_smearing_param (float, optional): The stout field smearing
            parameter to use before doing the inversion
          source_smear_type (str, optional): The type of smearing to apply
            to the source. Currently "jacobi" is the only supported smearing
            type.
          num_source_smears (int, optional): The number of Jacobi smears to apply
            to the source before inverting.
          source_smearing_param (float, optional): The Jacobi field smearing
            parameter to use before doing the inversion
          sink_smear_type (str, optional): The type of smearing to apply
            to the sink. Currently "jacobi" is the only supported smearing
            type.
          num_sink_smears (int, optional): The number of Jacobi smears to apply
            to the sink before inverting.
          sink_smearing_param (float, optional): The Jacobi field smearing
            parameter to use before doing the inversion
          solver_method (str, optional): The algorithm to use when doing the
            inversion. Currently "conjugate_gradient" and "bicgstab" are
            available.
          boundary_conditions (list, optional): The phase factors to apply to
            the fermion fields at the boundaries of the lattice.
          precondition (bool, optional): Determines whether the preconditioned
            form of the Dirac operator is used.
          max_iterations (int, optional): The maximum number of iterations the
            solver will perform before giving up.
          tolerance (float, optional): The maximum residual the solver will
            allow before considering convergence to have been reached.
          verbosity (int, optional): Determines how much inversion is outputted
            during the inversion. Values greater than one produce output, whilst
            1 or 0 will produce no output.
            
        Returns:
          Propagator: The propagator encapsulated in a Propagator object
          
        Examples:
          Create a lattice, thermalize it, and invert on a smeared source.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> prop = lattice.get_hamberwu_propagator(0.4, num_source_smears=2,
          ...                                      source_smearing_param=0.4)
        """
        
        if verbosity > 0:
            verbosity -= 1
        
        raw_propagator \
          = np.array(lattice.Lattice \
                     .get_hamberwu_propagator(
                         self, mass, source_site, num_field_smears,
                         field_smearing_param,
                         dicts.smearing_types[source_smear_type],
                         num_source_smears, source_smearing_param,
                         dicts.smearing_types[sink_smear_type], num_sink_smears,
                         sink_smearing_param, dicts.solver_methods[solver_method],
                         boundary_conditions, dicts.truefalse[precondition],
                         max_iterations, tolerance, verbosity))
        
        prop = np.swapaxes(np.reshape(raw_propagator, (self.T, self.L, self.L,
                                                       self.L, 4, 3, 4, 3)), 5,
                                                       6)
        
        out = propagator.Propagator(prop, self.L, self.T, self.beta, self.ut,
                                    self.us, self.chi, self.action, "hamber-wu",
                                    mass, {}, boundary_conditions, source_site,
                                    num_field_smears, field_smearing_param,
                                    source_smear_type, num_source_smears,
                                    source_smearing_param, sink_smear_type,
                                    num_sink_smears, sink_smearing_param)
        
        return out
    
    def get_naik_propagator(self, mass,
                            source_site=[0, 0, 0, 0],
                            num_field_smears=0,
                            field_smearing_param=1.0,
                            source_smear_type="jacobi",
                            num_source_smears=0,
                            source_smearing_param=1.0,
                            sink_smear_type="jacobi",
                            num_sink_smears=0,
                            sink_smearing_param=1.0,
                            solver_method="conjugate_gradient",
                            boundary_conditions = [-1, 1, 1, 1],
                            precondition=False,
                            max_iterations=1000,
                            tolerance=1e-8,
                            verbosity=0):
        """Compute the Naik propagator using the Naik fermion action
                       
        Args:
          mass (float): The bare quark mass
          source_site (list, optional): The source site to use when doing the
            inversion
          num_field_smears (int, optional): The number of stout field smears
            applied before doing the inversion
          field_smearing_param (float, optional): The stout field smearing
            parameter to use before doing the inversion
          source_smear_type (str, optional): The type of smearing to apply
            to the source. Currently "jacobi" is the only supported smearing
            type.
          num_source_smears (int, optional): The number of Jacobi smears to apply
            to the source before inverting.
          source_smearing_param (float, optional): The Jacobi field smearing
            parameter to use before doing the inversion
          sink_smear_type (str, optional): The type of smearing to apply
            to the sink. Currently "jacobi" is the only supported smearing
            type.
          num_sink_smears (int, optional): The number of Jacobi smears to apply
            to the sink before inverting.
          sink_smearing_param (float, optional): The Jacobi field smearing
            parameter to use before doing the inversion
          solver_method (str, optional): The algorithm to use when doing the
            inversion. Currently "conjugate_gradient" and "bicgstab" are
            available.
          boundary_conditions (list, optional): The phase factors to apply to
            the fermion fields at the boundaries of the lattice.
          precondition (bool, optional): Determines whether the preconditioned
            form of the Dirac operator is used.
          max_iterations (int, optional): The maximum number of iterations the
            solver will perform before giving up.
          tolerance (float, optional): The maximum residual the solver will
            allow before considering convergence to have been reached.
          verbosity (int, optional): Determines how much inversion is outputted
            during the inversion. Values greater than one produce output, whilst
            1 or 0 will produce no output.
            
        Returns:
          Propagator: The propagator encapsulated in a Propagator object
          
        Examples:
          Create a lattice, thermalize it, and invert on a smeared source.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> prop = lattice.get_naik_propagator(0.4, num_source_smears=2,
          ...                                    source_smearing_param=0.4)
        """
        
        if verbosity > 0:
            verbosity -= 1
        
        raw_propagator \
          = np.array(lattice.Lattice \
                     .get_naik_propagator(
                         self, mass, source_site, num_field_smears,
                         field_smearing_param,
                         dicts.smearing_types[source_smear_type],
                         num_source_smears, source_smearing_param,
                         dicts.smearing_types[sink_smear_type], num_sink_smears,
                         sink_smearing_param, dicts.solver_methods[solver_method],
                         boundary_conditions, dicts.truefalse[precondition],
                         max_iterations, tolerance, verbosity))
        
        prop = np.swapaxes(np.reshape(raw_propagator, (self.T, self.L, self.L,
                                                       self.L, 4, 3, 4, 3)), 5,
                                                       6)
        
        out = propagator.Propagator(prop, self.L, self.T, self.beta, self.ut,
                                    self.us, self.chi, self.action, "hamber-wu",
                                    mass, {}, boundary_conditions, source_site,
                                    num_field_smears, field_smearing_param,
                                    source_smear_type, num_source_smears,
                                    source_smearing_param, sink_smear_type,
                                    num_sink_smears, sink_smearing_param)
        
        return out
    
    def invert_wilson_dirac(self, eta, mass, boundary_conditions=[-1, 1, 1, 1],
                            solver_method="conjugate_gradient",
                            precondition=False, max_iterations=1000,
                            tolerance=1e-8, verbosity=0):
        """Inverts the Wilson Dirac operator on the specified lattice spinor
                            
        Args:
          eta (numpy.ndarray): The source on which to invert the Dirac operator.
            Must have shape (T, L, L, L, 4, 3).
          mass (float): The bare mass of the quark that the Dirac operator
            represents.
          boundary_conditions (list, optional): The phase factors to apply to
            the fermion fields at the boundaries of the lattice.
          solver_method (str, optional): The method to use to invert the
            Dirac operator.
          precondition (bool, optional): Determines whether preconditioning
            should be used when performing the inversion.
          max_iterations (int, optional): The maximum number of iterations
            the solver will perform before giving up.
          tolerance (float, optional): The maximum residual the solver will
            allow before considering convergence to have been reached.
          verbosity (int, optional): Determines how much inversion is outputted
            during the inversion. Values greater than one produce output, whilst
            1 or 0 will produce no output.
        
        Returns:
          numpy.ndarray: The spinor resulting from the inversion.
          
        Examples:
          Here we create a lattice and a source, then invert a Wilson Dirac
          operator on it.
          
          >>> import pyQCD
          >>> lat = pyQCD.Lattice()
          >>> import numpy as np
          >>> eta = np.zeros(lat.shape + (4, 3), dtype=np.complex)
          >>> eta[0, 0, 0, 0, 0] = 1.0
          >>> psi = lat.invert_wilson_dirac(eta, 0.4)
        """
        
        eta = eta.flatten()
        
        try:
            eta = eta.tolist()
        except AttributeError:
            pass
        
        psi = lattice.Lattice \
          .invert_wilson_dirac(self, eta, mass, boundary_conditions,
                               dicts.solver_methods[solver_method],
                               dicts.truefalse[precondition],
                               max_iterations, tolerance, verbosity)
        
        psi = np.array(psi)
        psi = psi.reshape((self.T, self.L, self.L, self.L, 4, 3))
        
        return psi
    
    def invert_hamberwu_dirac(self, eta, mass, boundary_conditions=[-1, 1, 1, 1],
                              solver_method="conjugate_gradient",
                              precondition=False, max_iterations=1000,
                              tolerance=1e-8, verbosity=0):
        """Inverts the Hamber-Wu Dirac operator on the specified lattice spinor
                            
        Args:
          eta (numpy.ndarray): The source on which to invert the Dirac operator.
            Must have shape (T, L, L, L, 4, 3).
          mass (float): The bare mass of the quark that the Dirac operator
            represents.
          boundary_conditions (list, optional): The phase factors to apply to
            the fermion fields at the boundaries of the lattice.
          solver_method (str, optional): The method to use to invert the
            Dirac operator.
          precondition (bool, optional): Determines whether preconditioning
            should be used when performing the inversion.
          max_iterations (int, optional): The maximum number of iterations
            the solver will perform before giving up.
          tolerance (float, optional): The maximum residual the solver will
            allow before considering convergence to have been reached.
          verbosity (int, optional): Determines how much inversion is outputted
            during the inversion. Values greater than one produce output, whilst
            1 or 0 will produce no output.
        
        Returns:
          numpy.ndarray: The spinor resulting from the inversion.
          
        Examples:
          Here we create a lattice and a source, then invert a Hamber-Wu Dirac
          operator on it.
          
          >>> import pyQCD
          >>> lat = pyQCD.Lattice()
          >>> import numpy as np
          >>> eta = np.zeros(lat.shape + (4, 3), dtype=np.complex)
          >>> eta[0, 0, 0, 0, 0] = 1.0
          >>> psi = lat.invert_hamberwu_dirac(eta, 0.4)
        """
        
        eta = eta.flatten()
        
        try:
            eta = eta.tolist()
        except AttributeError:
            pass
        
        psi = lattice.Lattice\
          .invert_hamberwu_dirac(self, eta, mass, boundary_conditions,
                                 dicts.solver_methods[solver_method],
                                 dicts.truefalse[precondition],
                                 max_iterations, tolerance, verbosity)
        
        psi = np.array(psi)
        psi = psi.reshape((self.T, self.L, self.L, self.L, 4, 3))
        
        return psi
    
    def invert_naik_dirac(self, eta, mass, boundary_conditions=[-1, 1, 1, 1],
                          solver_method="conjugate_gradient",
                          precondition=False, max_iterations=1000,
                          tolerance=1e-8, verbosity=0):
        """Inverts the Naik Dirac operator on the specified lattice spinor
                            
        Args:
          eta (numpy.ndarray): The source on which to invert the Dirac operator.
            Must have shape (T, L, L, L, 4, 3).
          mass (float): The bare mass of the quark that the Dirac operator
            represents.
          boundary_conditions (list, optional): The phase factors to apply to
            the fermion fields at the boundaries of the lattice.
          solver_method (str, optional): The method to use to invert the
            Dirac operator.
          precondition (bool, optional): Determines whether preconditioning
            should be used when performing the inversion.
          max_iterations (int, optional): The maximum number of iterations
            the solver will perform before giving up.
          tolerance (float, optional): The maximum residual the solver will
            allow before considering convergence to have been reached.
          verbosity (int, optional): Determines how much inversion is outputted
            during the inversion. Values greater than one produce output, whilst
            1 or 0 will produce no output.
        
        Returns:
          numpy.ndarray: The spinor resulting from the inversion.
          
        Examples:
          Here we create a lattice and a source, then invert a Hamber-Wu Dirac
          operator on it.
          
          >>> import pyQCD
          >>> lat = pyQCD.Lattice()
          >>> import numpy as np
          >>> eta = np.zeros(lat.shape + (4, 3), dtype=np.complex)
          >>> eta[0, 0, 0, 0, 0] = 1.0
          >>> psi = lat.invert_naik_dirac(eta, 0.4)
        """
        
        eta = eta.flatten()
        
        try:
            eta = eta.tolist()
        except AttributeError:
            pass
        
        psi = lattice.Lattice\
          .invert_naik_dirac(self, eta, mass, boundary_conditions,
                             dicts.solver_methods[solver_method],
                             dicts.truefalse[precondition],
                             max_iterations, tolerance, verbosity)
        
        psi = np.array(psi)
        psi = psi.reshape((self.T, self.L, self.L, self.L, 4, 3))
        
        return psi
    
    def invert_dwf_dirac(self, eta, mass, M5, Ls, kernel="wilson",
                         boundary_conditions=[-1, 1, 1, 1],
                         solver_method="conjugate_gradient",
                         precondition=False, max_iterations=1000,
                         tolerance=1e-8, verbosity=0):
        """Inverts the domain wall Dirac operator on the specified lattice spinor
        
        The domain wall (Shamir) operator is generalised and can use any existing
        four dimensional action that accepts a single mass parameter.
                            
        Args:
          eta (numpy.ndarray): The source on which to invert the Dirac operator.
            Must have shape (T, L, L, L, 4, 3).
          mass (float): The bare mass of the quark that the Dirac operator
            represents.
          M5 (float): The domain wall height.
          Ls (int): The extent of the fifth dimension.
          kernel (str, optional): The four dimensional kernel to use. Currently
            "wilson" and "hamber-wu" are supported.
          boundary_conditions (list, optional): The phase factors to apply to
            the fermion fields at the boundaries of the lattice.
          solver_method (str, optional): The method to use to invert the
            Dirac operator.
          precondition (bool, optional): Determines whether preconditioning
            should be used when performing the inversion.
          max_iterations (int, optional): The maximum number of iterations
            the solver will perform before giving up.
          tolerance (float, optional): The maximum residual the solver will
            allow before considering convergence to have been reached.
          verbosity (int, optional): Determines how much inversion is outputted
            during the inversion. Values greater than one produce output, whilst
            1 or 0 will produce no output.
        
        Returns:
          numpy.ndarray: The spinor resulting from the inversion.
          
        Examples:
          Here we create a lattice and a source, then invert a Hamber-Wu Dirac
          operator on it.
          
          >>> import pyQCD
          >>> lat = pyQCD.Lattice()
          >>> import numpy as np
          >>> eta = np.zeros(lat.shape + (4, 3), dtype=np.complex)
          >>> eta[0, 0, 0, 0, 0] = 1.0
          >>> psi = lat.invert_hamberwu_dirac(eta, 0.4)
        """
        
        eta = eta.flatten()
        
        try:
            eta = eta.tolist()
        except AttributeError:
            pass
        
        psi = lattice.Lattice \
          .invert_dwf_dirac(self, eta, mass, M5, Ls,
                            dicts.fermion_actions[kernel], boundary_conditions,
                            dicts.solver_methods[solver_method],
                            dicts.truefalse[precondition], max_iterations,
                            tolerance, verbosity)
        
        psi = np.array(psi)
        psi = psi.reshape((Ls, self.T, self.L, self.L, self.L, 4, 3))
        
        return psi
    
    def apply_wilson_dirac(self, psi, mass, boundary_conditions=[-1, 1, 1, 1],
                           precondition=False):
        """Applies the Wilson dirac operator to the specified lattice spinor
                           
        Args:
          psi (numpy.ndarray): The spinor to which the Dirac operator is
            applied. Must have shape (T, L, L, L, 4, 3).
          mass (float): The bare mass of the quark that the Dirac operator
            represents.
          boundary_conditions (list, optional): The phase factors to apply to
            the fermion fields at the boundaries of the lattice.
          precondition (bool, optional): Specifies whether the preconditioned
            form of the Dirac operator is applied to the spinor.
            
        Returns:
          numpy.ndarray: The spinor field resulting from the operator application
          
        Examples:
          Create a lattice and a spinor, then apply the Dirac operator to the
          spinor.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> import numpy as np
          >>> psi = np.zeros((lattice.T, lattice.L, lattice.L, lattice.L, 4, 3))
          >>> psi[0, 0, 0, 0, 0, 0] = 1.0
          >>> eta = lattice.apply_wilson_dirac(psi, 0.4)
          
          Create a lattice, then use the Dirac operator to create a scipy
          LinearOperator, which can be used to do inversions using the scipy
          sparse solvers.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> import numpy as np
          >>> import scipy.sparse.linalg as spla
          >>> N = 12 * lattice.T * lattice.L**3
          >>> matvec = lambda psi: \
          ...   lattice.apply_wilson_dirac(psi, 0.4).flatten(), 
          >>> wilson_dirac = spla.LinearOperator((N, N), matvec,
          ...                                    dtype=np.complex)
          >>> eta = np.zeros(N)
          >>> eta[0] = 1.0
          >>> psi = spla.gmres(wilson_dirac, eta)          
        """
        
        psi = psi.flatten()
        
        try:
            psi = psi.tolist()
        except AttributeError:
            pass
        
        eta = lattice.Lattice.apply_wilson_dirac(self, psi, mass,
                                                 boundary_conditions,
                                                 dicts.truefalse[precondition])
        
        eta = np.array(eta)
        eta = eta.reshape((self.T, self.L, self.L, self.L, 4, 3))
        
        return eta
    
    def apply_hamberwu_dirac(self, psi, mass, boundary_conditions=[-1, 1, 1, 1],
                             precondition=False):
        """Applies the Hamber-Wu dirac operator to the specified lattice spinor
                           
        Args:
          psi (numpy.ndarray): The spinor to which the Dirac operator is
            applied. Must have shape (T, L, L, L, 4, 3).
          mass (float): The bare mass of the quark that the Dirac operator
            represents.
          boundary_conditions (list, optional): The phase factors to apply to
            the fermion fields at the boundaries of the lattice.
          precondition (bool, optional): Specifies whether the preconditioned
            form of the Dirac operator is applied to the spinor.
            
        Returns:
          numpy.ndarray: The spinor field resulting from the operator application
          
        Examples:
          Create a lattice and a spinor, then apply the Dirac operator to the
          spinor.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> import numpy as np
          >>> psi = np.zeros((lattice.T, lattice.L, lattice.L, lattice.L, 4, 3))
          >>> psi[0, 0, 0, 0, 0, 0] = 1.0
          >>> eta = lattice.apply_hamberwu_dirac(psi, 0.4)
          
          Create a lattice, then use the Dirac operator to create a scipy
          LinearOperator, which can be used to do inversions using the scipy
          sparse solvers.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> import numpy as np
          >>> import scipy.sparse.linalg as spla
          >>> N = 12 * lattice.T * lattice.L**3
          >>> matvec = lambda psi: \
          ...   lattice.apply_hamberwu_dirac(psi, 0.4).flatten(), 
          >>> wilson_dirac = spla.LinearOperator((N, N), matvec,
          ...                                    dtype=np.complex)
          >>> eta = np.zeros(N)
          >>> eta[0] = 1.0
          >>> psi = spla.gmres(wilson_dirac, eta)          
        """
        
        psi = psi.flatten()
        
        try:
            psi = psi.tolist()
        except AttributeError:
            pass
        
        eta = lattice.Lattice.apply_hamberwu_dirac(self, psi, mass,
                                                   boundary_conditions,
                                                   dicts.truefalse[precondition])
        
        eta = np.array(eta)
        eta = eta.reshape((self.T, self.L, self.L, self.L, 4, 3))
        
        return eta
    
    def apply_naik_dirac(self, psi, mass, boundary_conditions=[-1, 1, 1, 1],
                         precondition=False):
        """Applies the Naik dirac operator to the specified lattice spinor
                           
        Args:
          psi (numpy.ndarray): The spinor to which the Dirac operator is
            applied. Must have shape (T, L, L, L, 4, 3).
          mass (float): The bare mass of the quark that the Dirac operator
            represents.
          boundary_conditions (list, optional): The phase factors to apply to
            the fermion fields at the boundaries of the lattice.
          precondition (bool, optional): Specifies whether the preconditioned
            form of the Dirac operator is applied to the spinor.
            
        Returns:
          numpy.ndarray: The spinor field resulting from the operator application
          
        Examples:
          Create a lattice and a spinor, then apply the Naik Dirac operator to the
          spinor.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> import numpy as np
          >>> psi = np.zeros((lattice.T, lattice.L, lattice.L, lattice.L, 4, 3))
          >>> psi[0, 0, 0, 0, 0, 0] = 1.0
          >>> eta = lattice.apply_naik_dirac(psi, 0.4)
          
          Create a lattice, then use the Dirac operator to create a scipy
          LinearOperator, which can be used to do inversions using the scipy
          sparse solvers.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> import numpy as np
          >>> import scipy.sparse.linalg as spla
          >>> N = 12 * lattice.T * lattice.L**3
          >>> matvec = lambda psi: \
          ...   lattice.apply_naik_dirac(psi, 0.4).flatten(), 
          >>> wilson_dirac = spla.LinearOperator((N, N), matvec,
          ...                                    dtype=np.complex)
          >>> eta = np.zeros(N)
          >>> eta[0] = 1.0
          >>> psi = spla.gmres(wilson_dirac, eta)          
        """
        
        psi = psi.flatten()
        
        try:
            psi = psi.tolist()
        except AttributeError:
            pass
        
        eta = lattice.Lattice.apply_hamberwu_dirac(self, psi, mass,
                                                   boundary_conditions,
                                                   dicts.truefalse[precondition])
        
        eta = np.array(eta)
        eta = eta.reshape((self.T, self.L, self.L, self.L, 4, 3))
        
        return eta
    
    def apply_dwf_dirac(self, psi, mass, M5, Ls, kernel="wilson",
                        boundary_conditions=[-1, 1, 1, 1], precondition=False):
        """Applies a generic DWF dirac operator to the specified 5D lattice
        spinor
                           
        Args:
          psi (numpy.ndarray): The spinor to which the Dirac operator is
            applied. Must have shape (T, L, L, L, 4, 3).
          mass (float): The bare mass of the quark that the Dirac operator
            represents.
          M5 (float): The domain wall height to use.
          Ls (int): The extent of the fifth dimension.
          kernel (str, optional): The four dimensional kernel to use. Only
            kernels with a single quark mass parameter are permissible here.
          boundary_conditions (list, optional): The phase factors to apply to
            the fermion fields at the boundaries of the lattice.
          precondition (bool, optional): Specifies whether the preconditioned
            form of the Dirac operator is applied to the spinor.
            
        Returns:
          numpy.ndarray: The 5D spinor field resulting from the operator
          application
          
        Examples:
          Create a lattice and a spinor, then apply the Dirac operator to the
          spinor.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> import numpy as np
          >>> psi = np.zeros((4, lattice.T, lattice.L, lattice.L, lattice.L,
          ...                 4, 3))
          >>> psi[0, 0, 0, 0, 0, 0, 0] = 1.0
          >>> eta = lattice.apply_dwf_dirac(psi, 0.4, 1.8, 4)     
        """
        
        psi = psi.flatten()
        
        try:
            psi = psi.tolist()
        except AttributeError:
            pass
        
        eta = lattice.Lattice.apply_dwf_dirac(self, psi, mass, M5, Ls,
                                              dicts.fermion_actions[kernel],
                                              boundary_conditions,
                                              dicts.truefalse[precondition])
        
        eta = np.array(eta)
        eta = eta.reshape((Ls, self.T, self.L, self.L, self.L, 4, 3))
        
        return eta
    
    def apply_jacobi_smearing(self, psi, num_smears, smearing_param,
                              boundary_conditions=[-1, 1, 1, 1]):
        """Applies the Jacobi smearing operator to the specified lattice spinor
                           
        Args:
          psi (numpy.ndarray): The spinor to which the Dirac operator is
            applied. Must have shape (T, L, L, L, 4, 3).
          num_smears (int): The number of Jacobi smears to apply to the supplied
            spinor.
          smearing_param (float): The Jacobi smearing parameter.
          boundary_conditions (list, optional): The phase factors to apply to
            the fermion fields at the boundaries of the lattice.
            
        Returns:
          numpy.ndarray: The spinor field resulting from the operator application
          
        Examples:
          Create a lattice and a spinor, then apply the smearing operator to the
          spinor.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> import numpy as np
          >>> psi = np.zeros((lattice.T, lattice.L, lattice.L, lattice.L, 4, 3))
          >>> psi[0, 0, 0, 0, 0, 0] = 1.0
          >>> eta = lattice.apply_wilson_dirac(psi, 0.4)
        """
        
        psi = psi.flatten()
        
        try:
            psi = psi.tolist()
        except AttributeError:
            pass
        
        eta = lattice.Lattice.apply_jacobi_smearing(self, psi, num_smears,
                                                    smearing_param)
        
        eta = np.array(eta)
        eta = eta.reshape((self.T, self.L, self.L, self.L, 4, 3))
        
        return eta
    
    def get_av_link(self):
        """Computes the mean of the real part of the trace of each link matrix
        
        Returns:
          float: The value of the mean link.
          
        Examples:
          Create a lattice and compute the mean link.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice()
          >>> lattice.get_av_link()
          1.0
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
        "Mean temporal link (ut): {}\n" \
        "Mean spatial link (us): {}\n" \
        "Anisotropy factor (chi): {}\n" \
        "Update method: {}\n" \
        "Measurement spacing: {}\n" \
        "Parallel updates: {}\n" \
        "Parallel update block size: {}\n" \
        "Random number seed: {}".format(self.L, self.T, self.L**3 * self.T,
                                        self.L**3 * self.T * 4, self.action,
                                        self.beta, self.ut, self.us, self.chi,
                                        self.update_method, self.num_cor,
                                        self.parallel_updates, self.block_size,
                                        self.rand_seed)
        
        return out
