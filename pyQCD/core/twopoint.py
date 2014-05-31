from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np

from .constants import *
from .propagator import spin_prod, prop_adjoint
from .dataset import _parmap
from .log import Log
    
def filter_correlators(data, label=None, masses=None, momentum=None,
                       source_type=None, sink_type=None):
    """Filters the dictionary of correlators returned by one of the CHROMA
    import functions. Returns the specified correlator, or a dictionary
    containing the correlators that match the arguments supplied to the
    function.
        
    Args:
      data (dict): The dictionary of correlators to be filtered
      label (str, optional): The correlator label
      masses (array-like, optional): The masses of the valence quarks that
        form the hadron that corresponds to the correlator.
      momentum (array-like, optional): The momentum of the hadron that
        corresponds to the correlator
      source_type (str, optional): The type of the source used when
        computing the propagator(s) used to compute the corresponding
        correlator.
      sink_type (str, optional): The type of the sink used when
        computing the propagator(s) used to compute the corresponding
        correlator.
            
    Returns:
      dict or numpy.ndarray: The correlator(s) matching the criteria
          
      If the supplied criteria match more than one correlator, then
      a dict is returned, containing the correlators that match these
      criteria. The keys are tuples containing the corresponding
      criteria for the correlators. If only one correlator is found, then
      the correlator itself is returned as a numpy array.
          
    Examples:
      Load a two-point object from disk and retreive the correlator
      denoted by the label "pion" with zero momentum.
          
      >>> import pyQCD
      >>> correlators = pyQCD.load_chroma_hadspec("correlators.dat.xml")
      >>> pyQCD.filter_correlators(correlators, "pion", momentum=(0, 0, 0))
      array([  9.19167425e-01,   4.41417607e-02,   4.22095090e-03,
               4.68472393e-04,   5.18833346e-05,   5.29751835e-06,
               5.84481783e-07,   6.52953123e-08,   1.59048703e-08,
               7.97830102e-08,   7.01262406e-07,   6.08545149e-06,
               5.71428481e-05,   5.05306201e-04,   4.74744759e-03,
               4.66148785e-02])
    """
        
    correlator_attributes = data.keys()
        
    if masses != None:
        masses = tuple([round(mass, 8) for mass in masses])
            
    if momentum != None:
        momentum = tuple(momentum)
        
    filter_params = [label, masses, momentum, source_type, sink_type]
        
    for i, param in enumerate(filter_params):       
        if param != None:
            correlator_attributes \
              = [attrib for attrib in correlator_attributes
                 if attrib[i] == param]
        
    if len(correlator_attributes) == 1:
        return data[correlator_attributes[0]]
    else:
        correlators = [data[attrib]
                       for attrib in correlator_attributes]
           
        return dict(zip(correlator_attributes, tuple(correlators)))

def compute_meson_corr(propagator1, propagator2, source_interpolator,
                       sink_interpolator, momenta=(0, 0, 0),
                       average_momenta=True, fold=False):
    """Computes the specified meson correlator
        
    Colour and spin traces are taken over the following product:
        
    propagator1 * source_interpolator * propagator2 * sink_interpolator
        
    Args:
      propagator1 (numpy.ndarray): The first propagator to use in calculating
        the correlator.
      propagator2 (numpy.ndarray): The second propagator to use in calculating
        the correlator.
      source_interpolator (numpy.ndarray or str): The interpolator describing
        the source of the two-point function. If a numpy array is passed, then
        it must have the shape (4, 4) (i.e. must encode some form of spin
        structure). If a string is passed, then the operator is searched for in
        pyQCD.constants.Gammas. A list of possible strings to use as this
        argument can be seen by calling pyQCD..available_interpolators()
      sink_interpolator (numpy.ndarray or str): The interpolator describing
        the sink of the two-point function. Conventions for this argument
        follow those of source_interpolator.
      momenta (list, optional): The momenta to project the spatial
        correlator onto. May be either a list of three ints defining a
        single lattice momentum, or a list of such lists defining multiple
        momenta.
      average_momenta (bool, optional): Determines whether the correlator
        is computed at all momenta equivalent to that in the momenta
        argument before an averable is taken (e.g. an average of the
        correlators for p = [1, 0, 0], [0, 1, 0], [0, 0, 1] and so on would
        be taken).
      fold (bool, optional): Determines whether the correlator is folded
        about it's mid-point.

    Returns:
      numpy.ndarray or dict: The computed correlator or correlators
            
    Examples:
      Create and thermalize a lattice, generate some propagators and use
      them to compute a pseudoscalar correlator.
          
      >>> import pyQCD
      >>> lattice = pyQCD.Lattice(4, 8, 5.5, "wilson", 10)
      >>> lattice.thermalize(100)
      >>> prop = lattice.get_wilson_propagator(0.4)
      >>> correlator = pyQCD.compute_meson_corr(prop, prop, "g5", "g5")
    """
        
    try:
        source_interpolator = Gammas[source_interpolator]
    except TypeError:
        pass
        
    try:
        sink_interpolator = Gammas[sink_interpolator]
    except TypeError:
        pass
    
    if type(momenta[0]) != list and type(momenta[0]) != tuple:
        momenta = [momenta]

    L = propagator1.shape[1]
    T = propagator1.shape[0]
        
    spatial_correlator = _compute_correlator(propagator1, propagator2,
                                             source_interpolator,
                                             sink_interpolator)
        
    mom_correlator = np.fft.fftn(spatial_correlator, axes=(1, 2, 3))

    out = {}
    
    # Now go through all momenta and compute the
    # correlators
    for momentum in momenta:
        if average_momenta:
            equiv_momenta = _get_all_momenta(momentum, L, T)
            # Put the equivalent momenta into a form so that we can filter
            # the relevant part of the mom space correlator out
            equiv_momenta = np.array(equiv_momenta)
            equiv_momenta = (equiv_momenta[:, 0],
                             equiv_momenta[:, 1],
                             equiv_momenta[:, 2])
            equiv_correlators \
              = np.transpose(mom_correlator, (1, 2, 3, 0))[equiv_momenta]
                    
            correlator = np.mean(equiv_correlators, axis=0)
                
        else:
            momentum = tuple([i % self.L for i in momentum])
            correlator = mom_correlator[:, momentum[0], momentum[1],
                                        momentum[2]]

        out[tuple(momentum)] = correlator

    if len(out.keys()) == 1:
        return list(out.values())[0]
    else:
        return out

def _get_all_momenta(p, L, T):
    """Generates all possible equivalent lattice momenta"""
        
    p2 = p[0]**2 + p[1]**2 + p[2]**2
        
    return [(px % L, py % L, pz % L)
            for px in range(-L // 2, L // 2)
            for py in range(-L // 2, L // 2)
            for pz in range(-L // 2, L // 2)
            if px**2 + py**2 + pz**2 == p2]

def _compute_correlator(prop1, prop2, gamma1, gamma2):
    """Calculates the correlator for all space-time points
    
    We're doing the following (g1 = gamma1, g2 = gamma2, p1 = prop1,
    p2 = prop2, g5 = gamma5):
        
    sum_{spin,colour} g1 * g5 * p1 * g5 * g2 * p2
    
    The g5s are used to find the Hermitian conjugate of the first propagator
    """
    
    gp1 = spin_prod(gamma1, prop_adjoint(prop1))
    gp2 = spin_prod(gamma2, prop2)
    
    return np.einsum('txyzijab,txyzjiba->txyz', gp1, gp2)

@Log("Computing all 256 meson correlators",
     ignore=("propagator1", "propagator2"))
def compute_meson_corr256(propagator1, propagator2, momenta=(0, 0, 0),
                          average_momenta=True, fold=False):
    """Computes and stores all 256 meson correlators within the
    current TwoPoint object. Labels akin to those in pyQCD.interpolators
    are used to denote the 16 gamma matrix combinations.
        
    Args:
      propagator1 (Propagator): The first propagator to use in calculating
        the correlator.
      propagator2 (Propagator): The second propagator to use in calculating
        the correlator.
      momenta (list, optional): The momenta to project the spatial
        correlator onto. May be either a list of three ints defining a
        single lattice momentum, or a list of such lists defining multiple
        momenta.
      average_momenta (bool, optional): Determines whether the correlator
        is computed at all momenta equivalent to that in the momenta
        argument before an averable is taken (e.g. an average of the
        correlators for p = [1, 0, 0], [0, 1, 0], [0, 0, 1] and so on would
        be taken).
      fold (bool, optional): Determines whether the correlator is folded
        about it's mid-point.
            
    Examples:
      Create and thermalize a lattice, generate some propagators and use
      them to compute a pseudoscalar correlator.
          
      >>> import pyQCD
      >>> lattice = pyQCD.Lattice(4, 8, 5.5, "wilson", 10)
      >>> lattice.thermalize(100)
      >>> prop = lattice.get_wilson_propagator(0.4)
      >>> correlator = pyQCD.compute_meson_corr256(prop, prop)
    """

    interp_pairs = [(Gamma1, Gamma2)
                    for Gamma1 in interpolators
                    for Gamma2 in interpolators]

    def parallel_function(gammas):
        return compute_meson_corr(propagator1, propagator2,
                                  gammas[0], gammas[1], momenta,
                                  average_momenta, fold)
    
    results = _parmap(parallel_function, interp_pairs)

    out = {}
    for interpolator, result in zip(interp_pairs, results):
        try:
            for mom, corr in result.items():
                out[(interpolator[0], interpolator[1], mom)] = corr

        except AttributeError:
            out[(interpolator[0], interpolator[1])] = result

    return out
