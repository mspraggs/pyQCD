import numpy as np
from observable import Observable

class Config(Observable):
    
    members = ['L', 'T', 'beta', 'u0', 'action']
    
    def __init__(self, links, L, T, beta, u0, action):
        """Create a field configuration container.
                 
        :param links: The field configuration
        :type links: :class:`np.ndarray` with shape :samp:`(T, L, L, L, 4, 3, 3)`
        :param L: The spatial extent of the corresponding :class:`Lattice`
        :type L: :class:`int`
        :param T: The temporal extent of the corresponding :class:`Lattice`
        :type T: :class:`int`
        :param beta: The inverse coupling
        :type beta: :class:`float`
        :param u0: The mean link
        :type u0: :class:`float`
        :param action: The gauge action
        :type action: :class:`str`, one of wilson, rectangle_improved or
        twisted_rectangle_improved
        :raises: ValueError
        """
        
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
        
        :param filename: The file to save to
        :type filename: :class:`str`
        """
        Observable.save(self, filename)
        
    @classmethod
    def load(cls, filename):
        """Loads and returns a configuration object from a numpy zip
        archive
        
        :param filename: The file to load from
        :type filename: :class:`str`
        """
        return super(Config, cls).load(filename)
    
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

