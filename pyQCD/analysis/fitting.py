from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import warnings

import numpy as np
import scipy.optimize as spop

def fold_correlator(correlator):
    """Folds the supplied correlator about it's mid-point.

    Args:
      correlator (numpy.ndarray): The correlator to be folded.
    
    Returns:
      numpy.ndarray: The folded correlator.

    Examples:
      Load a correlator from a numpy binary and fold it.

      >>> import numpy as np
      >>> import pyQCD
      >>> correlator = np.load("some_correlator.npy")
      >>> folded_correlator = pyQCD.fold_correlator(correlator)
    """

    if np.sign(correlator[1]) == np.sign(correlator[-1]):
        out = np.append(correlator[0], (correlator[:0:-1] + correlator[1:]) / 2)
    else:
        out = np.append(correlator[0], (correlator[1:] - correlator[:0:-1]) / 2)

    return out
            
def fit_correlators(correlator, fit_function, fit_range,
                   initial_parameters, correlator_std=None,
                   postprocess_function=None):
    """Fits the specified function to the supplied correlator(s) using
    scipy.optimize.leastsq
    
    Args:
      correlator (numpy.ndarray, list or dict): The correlator(s) to be fitted.
      fit_function (function): The function with which to fit the correlator.
        Must accept a list of fitting parameters as the first argument,
        followed by a numpy.ndarray of time coordinates, a numpy.ndarray, list,
        tuple or dictionary of correlator values, a numpy.ndarray of correlator
        errors and finally the fit range.
      fit_range (list or tuple): Specifies the timeslices over which to
        perform the fit. If a list or tuple with two elements is supplied,
        then range(*fit_range): is applied to the function to generate a
        list of timeslices to fit over.
      initial_parameters (list or tuple): The initial parameters to supply
        to the fitting routine.
      correlator_std (numpy.ndarray, optional): The standard deviation in
        the specified correlator.
      postprocess_function (function, optional): The function to apply to
        the result from scipy.optimize.leastsq.
                  
    Returns:
      list: The fitted parameters for the fit function.
            
    Examples:
      Load a correlator from disk and fit a simple exponential to it. A
      postprocess function to select the mass from the fit result is also
      specified.
    
      >>> import pyQCD
      >>> import numpy as np
      >>> correlator = np.load("my_correlator.npy")
      >>> def fit_function(b, t, Ct, err, fit_range):
      ...     x = t[fit_range]
      ...     y = Ct[fit_range]
      ...     yerr = err[fit_range]
      ...     return (y - b[0] * np.exp(-b[1] * x))
      ...
      >>> postprocess = lambda b: b[1]
      >>> pyQCD.fit_correlators(fit_function, [5, 10], [1., 1.],
      ...                       postprocess_function=postprocess)
      1.356389
      """
                                
    if len(fit_range) == 2:
        fit_range = range(*fit_range)

    try:
        t = np.arange(correlator.size)
    except AttributeError:
        try:
            t = np.arange(correlator[0].size)
        except KeyError:
            t = np.arange(correlator.values()[0].size)
        
    b, result = spop.leastsq(fit_function, initial_parameters,
                             args=(t, correlator, correlator_std, fit_range))
        
    if [1, 2, 3, 4].count(result) < 1:
        warnings.warn("fit failed when fitting correlator", RuntimeWarning)
        
    if postprocess_function == None:
        return b
    else:
        return postprocess_function(b)
            
def fit_1_correlator(correlator, fit_function, fit_range,
                   initial_parameters, correlator_std=None,
                   postprocess_function=None):
    """Fits the specified function to the given single correlator using
    scipy.optimize.leastsq
    
    Args:
      correlator (numpy.ndarray): The correlator to be fitted.
      fit_function (function): The function with which to fit the correlator.
        Must accept a list of fitting parameters as the first argument,
        followed by a numpy.ndarray of time coordinates, a numpy.ndarray of
        correlator values and a numpy.ndarray of correlator errors and finally
        the fit range.
      fit_range (list or tuple): Specifies the timeslices over which to
        perform the fit. If a list or tuple with two elements is supplied,
        then range(*fit_range): is applied to the function to generate a
        list of timeslices to fit over.
      initial_parameters (list or tuple): The initial parameters to supply
        to the fitting routine.
      correlator_std (numpy.ndarray, optional): The standard deviation in
        the specified correlator. If no standard deviation is supplied, then
        it is taken to be unity for each timeslice. This is equivalent to
        neglecting the error when computing the residuals for the fit.
      postprocess_function (function, optional): The function to apply to
        the result from scipy.optimize.leastsq.
                  
    Returns:
      list: The fitted parameters for the fit function.
            
    Examples:
      Load a correlator from disk and fit a simple exponential to it. A
      postprocess function to select the mass from the fit result is also
      specified.
    
      >>> import pyQCD
      >>> import numpy as np
      >>> correlator = np.load("my_correlator.npy")
      >>> def fit_function(b, t, Ct, err):
      ...     return (Ct - b[0] * np.exp(-b[1] * t)) / err
      ...
      >>> postprocess = lambda b: b[1]
      >>> pyQCD.fit_1_correlator(fit_function, [5, 10], [1., 1.],
      ...                        postprocess_function=postprocess)
      1.356389
      """
                
    if correlator_std == None:
        correlator_std = np.ones(correlator.size)
    if len(fit_range) == 2:
        fit_range = range(*fit_range)

    t = np.arange(correlator.size)

    x = t[list(fit_range)]
    y = correlator[list(fit_range)].real
    yerr = correlator_std[list(fit_range)].real
        
    b, result = spop.leastsq(fit_function, initial_parameters,
                             args=(x, y, yerr))
        
    if [1, 2, 3, 4].count(result) < 1:
        warnings.warn("fit failed when fitting correlator", RuntimeWarning)
        
    if postprocess_function == None:
        return b
    else:
        return postprocess_function(b)
        
def compute_energy(correlator, fit_range, initial_parameters,
                   correlator_std=None):
    """Computes the ground state energy of the specified correlator by fitting
    a curve to the data. The type of curve to be fitted (cosh or sinh) is
    determined from the shape of the correlator.
                     
    Args:
      correlator (numpy.ndarray): The correlator to be fitted.
      fit_range (list or tuple): Specifies the timeslices over which
        to perform the fit. If a list or tuple with two elements is
        supplied, then range(*fit_range): is applied to the function
        to generate a list of timeslices to fit over.
      initial_parameters (list or tuple): The initial parameters to supply
        to the fitting routine.
      correlator_std (numpy.ndarray, optional): The standard deviation
        in the specified correlator. If no standard deviation is supplied,
        then it is taken to be unity for each timeslice. This is equivalent
        to neglecting the error when computing the residuals for the fit.
        
    Returns:
      float: The fitted ground state energy.
          
    Examples:
      This function works in a very similar way to fit_correlator function,
      except the fitting function and the postprocessing function are already
      specified.
          
      >>> import pyQCD
      >>> import numpy as np
      >>> correlator = np.load("correlator.npy")
      >>> pyQCD.compute_energy(correlator, [5, 16], [1.0, 1.0])
      1.532435
    """

    T = correlator.size
                
    if np.sign(correlator[1]) == np.sign(correlator[-1]):

        def fit_function(b, t, Ct, err):
            return (Ct - b[0] * np.exp(-b[1] * t)
                    - b[0] * np.exp(-b[1] * (T - t))) / err
        
    else:

        def fit_function(b, t, Ct, err):
            return (Ct - b[0] * np.exp(-b[1] * T)
                    + b[0] * np.exp(-b[1] * (T - t))) / err
          
    postprocess_function = lambda b: b[1]
        
    return fit_1_correlator(correlator, fit_function, fit_range,
                            initial_parameters, correlator_std,
                            postprocess_function)
        
def compute_energy_sqr(correlator, fit_range, initial_parameters,
                       correlator_std=None):
    """Computes the square of the ground state energy of the specified
    correlator by fitting a curve to the data. The type of curve to be
    fitted (cosh or sinh) is determined from the shape of the correlator.
                     
    Args:
      correlator (numpy.ndarray): The correlator from which to extract
        the square energy
      fit_range (list): (list or tuple): Specifies the timeslices over
        which to perform the fit. If a list or tuple with two elements
        is supplied, then range(*fit_range): is applied to the function
        to generate a list of timeslices to fit over.
      initial_parameters (list or tuple): The initial parameters to
        supply to the fitting routine.
      correlator_std (numpy.ndarray, optional): The standard deviation
        in the specified correlator. If no standard deviation is
        supplied, then it is taken to be unity for each timeslice.
        This is equivalent to neglecting the error when computing
        the residuals for the fit.
      label (str, optional): The label of the correlator to be fitted.
        masses (list, optional): The bare quark masses of the quarks
        that form the hadron that the correlator corresponds to.
      momentum (list, optional): The momentum of the hadron that
        the correlator corresponds to.
      source_type (str, optional): The type of source used when
        generating the propagators that form the correlator.
      sink_type (str, optional): The type of sink used when
        generating the propagators that form the correlator.
        
    Returns:
      float: The fitted ground state energy squared.
          
    Examples:
      This function works in a very similar way to fit_correlator
      and compute_energy functions, except the fitting function and
      the postprocessing function are already specified.
    
      >>> import pyQCD
      >>> correlator = pyQCD.TwoPoint.load("correlator.npz")
      >>> correlator.compute_square_energy([5, 16], [1.0, 1.0])
      2.262435
    """
        
    return compute_energy(correlator, fit_range, initial_parameters,
                          correlator_std) ** 2
    
def compute_effmass(correlator, guess_mass=1.0):
    """Computes the effective mass for the supplied correlator by first
    trying to solve the ratio of the correlators on neighbouring time slices
    (see eq 6.57 in Gattringer and Lang). If this fails, then the function
    falls back to computing log(C(t) / C(t + 1)).
        
    Args:
      correlator (numpy.ndarray): The correlator used to compute the
        effective mass.
      guess_mass (float, optional): A guess effective mass to be used in
        the Newton method used to compute the effective mass.
            
    Returns:
      numpy.ndarray: The effective mass.
            
    Examples:
      Load a TwoPoint object containing a single correlator and compute
      its effective mass.
          
      >>> import pyQCD
      >>> import numpy as np
      >>> correlator = np.load("mycorrelator.npy")
      >>> pyQCD.compute_effmass(correlator)
          array([ 0.44806453,  0.41769303,  0.38761196,  0.3540968 ,
                  0.3112345 ,  0.2511803 ,  0.16695767,  0.05906789,
                 -0.05906789, -0.16695767, -0.2511803 , -0.3112345 ,
                 -0.3540968 , -0.38761196, -0.41769303, -0.44806453])
    """
    
    T = correlator.size
        
    try:
        if np.sign(correlator[1]) == np.sign(correlator[-1]):
            solve_function \
              = lambda m, t: np.cosh(m * (t - T / 2)) \
              / np.cosh(m * (t + 1 - T / 2))
        else:
            solve_function \
              = lambda m, t: np.sinh(m * (t - T / 2)) \
              / np.sinh(m * (t + 1 - T / 2))
          
        ratios = correlator / np.roll(correlator, -1)
        effmass = np.zeros(T)
            
        for t in range(T):
            function = lambda m: solve_function(m, t) - ratios[t]
            effmass[t] = spop.newton(function, guess_mass, maxiter=1000)
                
        return effmass
        
    except RuntimeError:        
        return np.log(np.abs(correlator / np.roll(correlator, -1)))
