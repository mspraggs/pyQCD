from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import logging

import numpy as np

from .constants import gamma5

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

def compute_propagator(src_func, invert_func, src_smear=None, snk_smear=None):
    """Extensible propagator generation function

    Args:
      src_func (function): A function used to generate the source used by the
        inversion function. This function must accept exactly two arguments.
        The first should be the spin index of the source and the second the
        colour index of the source. The resulting source should have shape
        (T, L, L, L, 4, 3).
      invert_func (function): The function used to perform the inversion. This
        function must accept exactly one argument - the source of shape
        (T, L, L, L, 4, 3). The function should return either the solution or
        a tuple containing the solution, the number of iterations performed by
        the solver, the final residual and the CPU time.
      src_smear (function, optional): The smearing function to apply to the
        source before inverting. This should accept one argument only, the
        source to smear, and return the smeared source. Both variables should
        have shape (T, L, L, L, 4, 3).
      snk_smear (function, optional): The smearing function to apply to the
        sink after inverting. This should accept one argument only, the sink
        to smear, and return the smeared source. Both variables should have
        shape (T, L, L, L, 4, 3).

    Returns:
      numpy.ndarray: The propagator as a numpy ndarray

    Examples:
      Create a lattice, then compute the Wilson propagator at tree level. Since
      the smearing functions are ommitted, no smearing is performed.

      >>> import pyQCD
      >>> lattice = pyQCD.Lattice(4, 8, 5.5, "wilson", 10)
      >>> make_source = lambda s, c: lattice.point_source(s, c, [0, 0, 0, 0])
      >>> inverter = lambda eta: lattice.invert_wilson_dirac(eta, 0.4)
      >>> prop = pyQCD.compute_propagator(make_source, inverter)

      Here we do some smearing too:

      >>> import pyQCD
      >>> lattice = pyQCD.Lattice(4, 8, 5.5, "wilson", 10)
      >>> make_source = lambda s, c: pyQCD.point_source(s, c, [0, 0, 0, 0])
      >>> smear_func = lambda psi: pyQCD.apply_jacobi_smearing(psi, 2, 1.5)
      >>> inverter = lambda eta: lattice.invert_wilson_dirac(eta, 0.4)
      >>> prop = pyQCD.compute_propagator(make_source, inverter,
      ...                                 smear_func, smear_func)
    """

    prop_log = logging.getLogger("propagator")
    invert_log = logging.getLogger("propagator.invert")

    spinor_shape = src_func(0, 0).shape

    propagator = np.zeros((4, 3) + spinor_shape, dtype=np.complex)

    prop_log.info("Started computing propagator")
    
    for spin in range(4):
        for colour in range(3):
            prop_log.info("Inverting for spin {} and colour {}"
                         .format(spin, colour))
            
            prop_log.info("Generating source")
            source = src_func(spin, colour)
            
            if src_smear != None:
                prop_log.info("Smearing source")
                source = src_smear(source)

            invert_log.info("Now inverting...")
            result = invert_func(source)
            if type(result) == tuple:
                solution = result[0]
                invert_log.info("Inversion finished after {} iterations "
                                "with residual {}".format(result[1], result[2]))
                invert_log.info("CPU time for this inversion: {} seconds"
                                .format(result[3]))
            else:
                solution = result
                invert_log.info("Done!")

            if snk_smear != None:
                prop_log.info("Smearing sink")
                solution = snk_smear(solution)

            propagator[spin, colour] = solution

    prop_log.info("Finished computing propagator")

    # This is a bit ugly, but I can't see another way to do this for an
    # arbitrary shaped propagator (e.g. 5D).

    shape_size = len(propagator.shape)
    # We're looking to construct the permuted ordering of the propagator
    # indices so we can put the spin and colour indices at the end. E.g.:
    # (4, 3, T, L, L, L, 4, 3) -> (T, L, L, L, 4, 4, 3, 3)
    transposed_order = (tuple(range(2, shape_size - 2))
                        + (shape_size - 2, 0, shape_size - 1, 1))

    return np.transpose(propagator, transposed_order)
