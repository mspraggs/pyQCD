import warnings

import numpy as np
import scipy.optimize as spop

from observable import Observable
    
def lattice_spacing(data):
    """Compute the lattice spacing using the Sommer scale

    Args:
      data (numpy.ndarray): The expectation values of all possible Wilson loops
        on the lattice. Should have shape (T, L), with element (i, j)
        corresponding to expectation value of a wilson loop with spatial extent
        j and temporal extent i.
        
    Returns:
      list: Contains both the lattice spacing and its inverse in fm and GeV
    """
        
    potential_params = _potential_parameters(data)
        
    spacing = 0.5 / np.sqrt((1.65 + potential_params[1])
                            / potential_params[0])
    return [spacing, 0.197 / spacing]
    
def pair_potential(data):
    """Computes the pair potential for a set of Wilson loops

    Args:
      data (numpy.ndarray): The expectation values of all possible Wilson loops
        on the lattice. Should have shape (T, L), with element (i, j)
        corresponding to expectation value of a wilson loop with spatial extent
        j and temporal extent i.
    
    Returns:
      numpy.ndarray: The pair potential, shape = (L,)
    """
    
    out = np.zeros(data.shape[0])
    t = np.arange(data.shape[1])
    
    fit_function = lambda b, t, W: W - b[0] * np.exp(-b[1] * t)
    
    for r in xrange(data.shape[0]):
        params, result = spop.leastsq(fit_function, [1., 1.],
                                      args=(t, data[r]))
    
    if [1, 2, 3, 4].count(result) < 1:
        warnings.warn("fit failed when calculating potential at "
                      "r = {}".format(r), RuntimeWarning)
                
    out[r] = params[1]
            
    return out
    
def _potential_parameters(data):
    """Fits the potential to extract the three fitting parameters"""
    
    potential = pair_potential(data)
    r = np.arange(potential.size)
    
    fit_function = lambda b, r, V: V - b[0] * r + b[1] / r + b[2]
        
    b, result = spop.leastsq(fit_function, [1., 1., 1.],
                             args=(r[1:], potential[1:]))
        
    if [1, 2, 3, 4].count(result) < 1:
        warnings.warn("fit failed when calculating potential at "
                      "r = {}".format(r), RuntimeWarning)
    
    return b
