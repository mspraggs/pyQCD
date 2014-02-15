import numpy as np
from observable import Observable

class Config(Observable):
    """Lattice field configuration container.
    
    Contains the data for one SU(3) field configuration, along
    with information relating to the action used to generate
    the configuration. The Config class also provides methods
    to save and load field configuration data (inherited from
    the Observable base class).
    
    Attributes:
      members (list): List of member variable names that will
        be stored on disk when the Config object is saved.
      L (int): The spatial extent of the lattice on which the
        the field configuration resides.
      T (int): The temporal extent of the lattice on which the
        the field configuration resides.
      beta (float): The inverse coupling used in the gauge action
        used to generate the field configuration.
      u0 (float): The mean link/tadpole coefficient used in the
        gauge action used to generate the field configuration.
      action (str): The gauge action used to generate the
        configuration.
      data (numpy.ndarray): The field configuration data in a
        numpy array of shape (T, L, L, L, 4, 3, 3).
     
    Args:
      links (np.ndarray): The field configuration, with shape
        (T, L, L, L, 4, 3, 3)
      L (int): The spatial extent of the lattice
      T (int): The temporal extent of the lattice
      beta (float): The inverse coupling
      u0 (float): The mean link/tadpole coefficient
      action (str): The gauge action
      
    Returns:
      Config: The field configuration object.
    
    Raises:
      ValueError: Shape of specified links array does not match the
        specified lattice extents
      
    Examples:
      Generate a set of gauge (non-SU(3)) links on a 4 ^ 3 x 8 lattice
      and use them to create a lattice object. (The 4, 3, 3
      components of the links shape are for the four space-time
      dimensions and the components of the SU(3) gauge field
      matrices.)
      
      >>> import pyQCD
      >>> import numpy
      >>> links = numpy.zeros((8, 4, 4, 4, 4, 3, 3))
      >>> config = pyQCD.Config(links, 4, 8, 5.5, 1.0, "wilson")
      
      Note instead of wilson one could have rectangle_improved or
      twisted_rectangle_improved
    """
    
    members = ['L', 'T', 'beta', 'u0', 'action']
    
    def __init__(self, links, L, T, beta, u0, action):
        """Constructor for pyQCD.Config (see help(pyQCD.Config)))"""
        # Validate the shape of the links array
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
        """Saves the configuration to a numpy zip archive
        
        Args:
          filename (str): The file to save to
          
        Examples:
          Extract the gauge field from a lattice object, extract the gauge
          field and write it to disk.
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice(8, 16, action="rectangle_improved")
          >>> lattice.thermalize(100)
          >>> config = lattice.get_config()
          >>> config.save("8c16_quenched_rectangle_symanzik")
        """
        Observable.save(self, filename)
        
    @classmethod
    def load(cls, filename):
        """Loads and returns a configuration object from a numpy zip
        archive
        
        Args:
          filename (str): The file to load from
        
        Returns:
          Config: The loaded configuration object
          
        Examples:
          Load a gauge field configuration from disk and set the corresponding
          gauge field of a lattice object to the new config
          
          >>> import pyQCD
          >>> lattice = pyQCD.Lattice(8, 16, action="rectangle_improved")
          >>> config = pyQCD.DataSet.load("8c16_quenched_rectangle_symanzik.npz")
          >>> lattice.set_config(config)
        """
        return super(Config, cls).load(filename)
    
    def __str__(self):
        
        out = \
          "Field Configuration Object\n" \
        "--------------------------\n" \
        "Spatial extent: {}\n" \
        "Temportal extent: {}\n" \
        "Gauge action: {}\n" \
        "Inverse coupling (beta): {}\n" \
        "Mean link (u0): {}".format(self.L, self.T, self.action, self.beta,
                                    self.u0)
        
        return out

