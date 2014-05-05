import numpy as np

from observable import Observable
from constants import gamma5

def prop_adjoint(propagator):
    """Returns the spin and colour adjoint of the given propagator
    
    Args:
      propagator (numpy.ndarray): The propagator of which to take the adjoint.
        Should have shape (T, L, L, L, 4, 4, 3, 3)
    
    Returns:
      numpy.ndarray: The adjoint of the specified propagator
      
    Examples:
      Load a propagator from disk and use it to compute the correlation
      function at avery site on the lattice. Here we use the adjoint
      function to compute the adjoint propagator.
      
      >>> import numpy as np
      >>> g5 = pyQCD.constants.gamma5
      >>> g0 = pyQCD.constants.gamma0
      >>> interpolator = np.dot(g0, g5)
      >>> prop = np.load("myprop.npy")
      >>> prop_adjoint = pyQCD.prop_adjoint(prop)
      >>> first_product = pyQCD.spin_prod(interpolator, prop_adjoint)
      >>> second_product = pyQCD.spin_prod(interpolator, prop)
      >>> correlator = np.einsum('txyzijab,txyzjiab->txyz',
      ...                        first_product, second_product)
    """
    
    out = np.transpose(propagator, (0, 1, 2, 3, 5, 4, 7, 6))
    out = np.conj(out)
    
    out = spin_prod(gamma5, out)
    out = spin_prod(out, gamma5)
    
    return out

def spin_prod(a, b):
    """Contracts the spin indices of the supplied propagator and gamma matrix
    
    The multiplication will be performed in the order the arguments are supplied,
    so a propagator can be left or right multiplied by a gamma matrix
    
    Args:
      a (numpy.ndarray): A propagator or gamma matrix. Shape should be
        (T, L, L, L, 4, 4, 3, 3) or (4, 4), respectively.
      b (numpy.ndarray): A propagator or gamma matrix (should not be the same one
        as a). Shape should be (T, L, L, L, 4, 4, 3, 3) or (4, 4), respectively.
        
    Returns:
      numpy.ndarray: The propagator, with the spin structure applied.
      
    Examples:
      Here we load a propagator then left multiply it by gamma 5.
      
      >>> import numpy as np
      >>> import pyQCD
      >>> prop = np.load("some_prop.npy")
      >>> prop_g5 = pyQCD.spin_prod(prop, pyQCD.gamma5)
    """
    
    try:
        # Left multiplication
        out = np.tensordot(a, b, (5, 0))
        out = np.transpose(out, (0, 1, 2, 3, 4, 7, 5, 6))
        
        return out
    
    except IndexError:
        # Right multiplication
        out = np.tensordot(a, b, (1, 4))
        out = np.transpose(out, (1, 2, 3, 4, 0, 5, 6, 7))
        
        return out
