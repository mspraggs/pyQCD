from observable import Observable
import numpy as np
import scipy.optimize as spop

class WilsonLoops(Observable):
    
    members = ['L', 'T', 'beta', 'u0', 'action', 'num_field_smears',
               'field_smearing_param']
    
    def __init__(self, loops, L, T, beta, u0, action,
                 num_field_smears, field_smearing_param):
        """Create a Wilson loop object
        
        :param loops: The wilson loops
        :type links: :class:`np.ndarray` with shape :samp:`(T-1, L-1)`
        :param L: The spatial extent of the corresponding :class:`Lattice`
        :type L: :class:`int`
        :param T: The temporal extent of the corresponding :class:`Lattice`
        :type T: :class:`int`
        :param beta: The inverse coupling
        :type beta: :class:`float`
        :param u0: The mean link
        :type u0: :class:`float`
        :param action: The gauge action
        :type action: :class:`str`, one of wilson, rectangle_improved or twisted_rectangle_improved
        :type source_site: :class:`list`
        :param num_field_smears: The number of stout smears applied when computing the propagator
        :type num_field_smears: :class:`int`
        :returns: :class:`WilsonLoops`
        :raises: ValueError
        """
        
        expected_shape = (L, T)
        
        if loops.shape != expected_shape:
            raise ValueError("Shape of specified Wilson loop array, {}, does "
                             "not match the specified lattice extents, {}"
                             .format(loops.shape, expected_shape))
        
        self.L = L
        self.T = T
        self.beta = beta
        self.u0 = u0
        self.action = action
        self.num_field_smears = num_field_smears
        self.field_smearing_param = field_smearing_param
        
        self.data = loops
    
    def lattice_spacing(self):
        """Compute the lattice spacing using the Sommer scale
        
        :returns: :class:`list` containg the lattice spacing in fm and the inverse lattice spacing in GeV
        """
        
        potential_params = self._potential_parameters()
        
        spacing = 0.5 / np.sqrt((1.65 + potential_params[1])
                                / potential_params[0])
        return [spacing, 0.197 / spacing]
    
    def pair_potential(self):
        """Computes the pair potential for the set of Wilson loops
        
        :returns: :class:`numpy.ndarray`
        """
        out = np.zeros(self.data.shape[0])
        t = np.arange(self.data.shape[1])
        
        fit_function = lambda b, t, W: W - b[0] * np.exp(-b[1] * t)
        
        for r in xrange(self.data.shape[0]):
            params, result = spop.leastsq(fit_function, [1., 1.],
                                          args=(t, self.data[r]))
            
            if [1, 2, 3, 4].count(result) < 1:
                print("Warning: fit failed when calculating potential at "
                      "r = {}".format(r))
                
            out[r] = params[1]
            
        return out
    
    def _potential_parameters(self):
        """Fits the potential to extract the three fitting parameters"""
        
        potential = self.pair_potential()
        r = np.arange(potential.size)
        
        fit_function = lambda b, r, V: V - b[0] * r + b[1] / r + b[2]
        
        b, result = spop.leastsq(fit_function, [1., 1., 1.],
                                 args=(r[1:], potential[1:]))
        
        if [1, 2, 3, 4].count(result) < 1:
            print("Warning: fit failed while determining potential parameters")
            
        return b

    def __str__(self):
        
        out = \
          "Wilson Loop Object\n" \
        "-----------------\n" \
        "Spatial extent: {}\n" \
        "Temportal extent: {}\n" \
        "Gauge action: {}\n" \
        "Inverse coupling (beta): {}\n" \
        "Mean link (u0): {}\n" \
        "Number of stout field smears: {}\n" \
        "Stout smearing parameter: {}\n" \
        .format(self.L, self.T, self.action, self.beta,
                self.u0, self.num_field_smears, self.field_smearing_param)
        
        return out
