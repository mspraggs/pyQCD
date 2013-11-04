import numpy as np
from observable import Observable

class Propagator(Observable):
    
    members = ['L', 'T', 'beta', 'u0', 'action', 'mass',
               'source_site', 'num_field_smears',
               'field_smearing_param', 'num_source_smears',
               'source_smearing_param', 'num_sink_smears',
               'sink_smearing_param']
        
    def __init__(self, propagator, L, T, beta, u0, action, mass, source_site,
                 num_field_smears, field_smearing_param, num_source_smears,
                 source_smearing_param, num_sink_smears, sink_smearing_param):
        """Create a propagator container.
                 
        :param propagator: The propagator array
        :type links: :class:`np.ndarray` with shape
        :samp:`(T, L, L, L, 4, 4, 3, 3)`
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
        :param mass: The bare quark mass
        :type mass: :class:`float`
        :param source_site: The source site used when doing the inversion
        :type source_site: :class:`list`
        :param num_field_smears: The number of stout smears applied when
        computing the propagator
        :type num_field_smears: :class:`int`
        :param field_smearing_param: The stout smearing parameter
        :type field_smearing_param: :class:`float`
        :param num_source_smears: The number of Jacobian smears performed
        on the source when computing the propagator
        :type num_source_smears: :class:`int`
        :param source_smearing_param: The Jacobi smearing parameter used
        when smearing the source
        :type source_smearing_param: :class:`float`
        :param num_sink_smears: The number of Jacobian smears performed
        on the source when computing the propagator
        :type num_sink_smears: :class:`int`
        :param sink_smearing_param: The Jacobi smearing parameter used
        when smearing the source
        :type sink_smearing_param: :class:`float`
        :raises: ValueError
        """
        
        expected_shape = (T, L, L, L, 4, 4, 3, 3)
        
        if propagator.shape != expected_shape:
            raise ValueError("Shape of specified propagator array, {}, does not "
                             "match the specified lattice extents, {}"
                             .format(propagator.shape, expected_shape))
        
        self.L = L
        self.T = T
        self.beta = beta
        self.u0 = u0
        self.action = action
        self.mass = mass
        self.source_site = source_site
        self.num_field_smears = num_field_smears
        self.field_smearing_param = field_smearing_param
        self.num_source_smears = num_source_smears
        self.source_smearing_param = source_smearing_param
        self.num_sink_smears = sink_smearing_param
        self.sink_smearing_param = sink_smearing_param
        
        self.data = propagator
    
    def save(self, filename):
        """Saves the propagator to a numpy zip archive
        
        :param filename: The file to save to
        :type filename: :class:`str`
        """
        Observable.save(self, filename)
        
    @classmethod
    def load(cls, filename):
        """Loads and returns a propagator object from a numpy zip
        archive
        
        :param filename: The file to load from
        :type filename: :class:`str`
        :returns: :class:`Propagator`
        """
        return super(Propagator, cls).load(filename)
    
    def __mul__(self, matrix):
        
        if type(matrix) != np.ndarray:
            raise ValueError("Error: Propagator cannot be multiplied by type "
                             "{}".format(type(matrix)))
        
        properties = dict(zip(Propagator.members,
                              [getattr(self, m) for m in Propagator.members]))
        
        if matrix.shape == (4, 4): # Multiply by a spin matrix
            out = np.tensordot(self.data, matrix, (5, 0))
            out = np.swapaxes(np.swapaxes(out, 6, 7), 5, 6)
            
            return Propagator(out, **properties)
        
        if matrix.shape == (3, 3): # Multiply by a colour matrix
            out = np.tensordot(self.data, matrix, (7, 0))
            
            return Propagator(out, **properties)
        
    def __repr__(self):
        
        out = \
          "Propagator Object\n" \
        "-----------------\n" \
        "Spatial extent: {}\n" \
        "Temportal extent: {}\n" \
        "Gauge action: {}\n" \
        "Inverse coupling (beta): {}\n" \
        "Mean link (u0): {}\n" \
        "Bare quark mass (m): {}\n" \
        "Inversion source site: {}\n" \
        "Number of stout field smears: {}\n" \
        "Stout smearing parameter: {}\n" \
        "Number of source Jacobi smears: {}\n" \
        "Source Jacobi smearing parameter: {}\n" \
        "Number of sink Jacobi smears: {}\n" \
        "Sink Jacobi smearing parameter: {}\n" \
        .format(self.L, self.T, self.action, self.beta,
                self.u0, self.mass, self.source_site,
                self.num_field_smears, self.field_smearing_param,
                self.num_source_smears, self.source_smearing_param,
                self.num_sink_smears, self.sink_smearing_param)
        
        return out
