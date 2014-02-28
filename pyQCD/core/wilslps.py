import warnings

import numpy as np
import scipy.optimize as spop

from observable import Observable

class WilsonLoops(Observable):
    """Create a Wilson loop object
    
    Args:
      loops (np.ndarray): The expectation values of all possible Wilson
        loops on the lattice. Should have shape (T, L), with element
        (i, j) representing a Wilson loop with spatial extent j and
        temporal extent i
      L (int): The spatial extent of the corresponding lattice
      T (int): The temporal extent of the corresponding lattice
      beta (float): The inverse coupling of the gauge action used
        to generate the configuration on which the Wilson loops are
        calculated
      u0 (float): The mean link/tadpole improvement coefficient
      action (str): The gauge action. If WilsonLoops are derived from
        a Lattice object, then this will correspond to the action used
        by the Lattice object.
      num_field_smears (int): The number of stout field smears applied to the
        gauge field before computing the Wilson loops
      field_smearing_parameter (float): The stout field smearing parameter
        used to stout smear the gauge field prior to computing the Wilson
        loops.
        
    Returns:
      WilsonLoops: The computed Wilson loops object
      
    Raises:
      ValueError: Shape of specified Wilson loop array does not match specified
        lattice extents.
      
    Examples:
      Create a Lattice object, thermalize, then compute the set of all
      Wilson loops, smearing the fields in the process.
      
      >>> import pyQCD
      >>> lattice = pyQCD.Lattice()
      >>> lattice.thermalize(100)
      >>> wilslps = lattice.get_wilson_loops(2, 0.4)
    """
    
    members = ['L', 'T', 'beta', 'u0', 'action', 'num_field_smears',
               'field_smearing_param']
    
    def __init__(self, loops, L, T, beta, u0, action,
                 num_field_smears, field_smearing_param):
        """Constructor for pyQCD.WilsonLoops (see help(pyQCD.WilsonLoops))"""
        
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
        
        Returns:
          list: Contains both the lattice spacing and its inverse
          
          The spacing is in fm and the inverse spacing is in GeV
        """
        
        potential_params = self._potential_parameters()
        
        spacing = 0.5 / np.sqrt((1.65 + potential_params[1])
                                / potential_params[0])
        return [spacing, 0.197 / spacing]
    
    def pair_potential(self):
        """Computes the pair potential for the set of Wilson loops
        
        Returns:
          numpy.ndarray: The pair potential, shape = (L,)
        """
        out = np.zeros(self.data.shape[0])
        t = np.arange(self.data.shape[1])
        
        fit_function = lambda b, t, W: W - b[0] * np.exp(-b[1] * t)
        
        for r in xrange(self.data.shape[0]):
            params, result = spop.leastsq(fit_function, [1., 1.],
                                          args=(t, self.data[r]))
            
            if [1, 2, 3, 4].count(result) < 1:
                warnings.warn("fit failed when calculating potential at "
                              "r = {}".format(r), RuntimeWarning)
                
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
            warnings.warn("fit failed when calculating potential at "
                          "r = {}".format(r), RuntimeWarning)
            
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
