from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import logging

import numpy as np

from .constants import gamma5
from .log import _logger, Log

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

@Log("Computing propagator...",
     ("src_template", "invert_func", "src_smear", "snk_smear"))
def compute_propagator(src_template, invert_func, src_smear=None,
                       snk_smear=None):
    """Extensible propagator generation function

    Args:
      src_template (numpy.ndarray): A source template used to generate the
        source used by the inversion function. This array should have the shape
        (T, L, L, L). This template is then used to generate the source for
        each spin and colour combination.
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

    logger = _logger()

    spinor_shape = src_template.shape
    propagator = np.zeros(spinor_shape + (4, 4, 3, 3), dtype=np.complex)

    for spin in range(4):
        for colour in range(3):
            logger.info("Inverting for spin {} and colour {}"
                         .format(spin, colour))
            
            source = np.zeros(spinor_shape + (4, 3), np.complex)
            source[..., spin, colour] = src_template
            
            if src_smear != None:
                source = src_smear(source)

            result = invert_func(source)
            solution = result[0] if type(result) == tuple else result

            if snk_smear != None:
                solution = snk_smear(solution)

            propagator[..., spin, :, colour] = solution

    logger.info("Finished computing propagator")

    return propagator

@Log("Smearing propagator", ("propagator", "smear_func"))
def smear_propagator(propagator, smear_func):
    """Applies the supplied smearing function to the supplied propagator

    Args:
      propagator (numpy.ndarray): The propagator to smear, with a shape of the
        form (..., 4, 4, 3, 3).
      smear_func (function): The smearing function to apply. Should be able
        to accept one of the 12 spin-colour components of the propagator,
        accepting a numpy ndarray of shape (..., 4, 3) and returning an array
        with the same shape.

    Returns:
      numpy.ndarray: The smeared propagator

    Examples:
      Create a lattice, generate a propagator then smear the propagator.

      >>> import pyQCD
      >>> lattice = pyQCD.Lattice(4, 8, 5.5, "wilson", 10)
      >>> invert_func = lambda psi: lattice.invert_wilson_dirac(psi, 0.4)
      >>> prop = pyQCD.compute_propagator(pyQCD.point_source([0, 0, 0, 0]),
      ...                                 invert_func)
      >>> smeared_prop = pyQCD.smear_propagator(prop)
    """

    new_prop = np.zeros(propagator.shape, dtype=np.complex)
    
    for alpha in range(4):
        for a in range(3):
            new_prop[..., alpha, :, a] \
              = smear_func(propagator[:, :, :, :, :, alpha, :, a])
            
    return new_prop
