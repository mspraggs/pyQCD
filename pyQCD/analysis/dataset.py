from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import warnings
import multiprocessing as mp

import numpy as np
import numpy.random as npr

# Following two functions taken from Stack Overflow answer by
# klaus se at http://stackoverflow.com/questions/3288595/ \
# multiprocessing-using-pool-map-on-a-function-defined-in-a-class
def _spawn(f):
    def fun(q_in,q_out):
        while True:
            i,x = q_in.get()
            if i is None:
                break
            q_out.put((i,f(x)))
    return fun

def _parmap(f, X, nprocs = mp.cpu_count()):
    q_in = mp.Queue(1)
    q_out = mp.Queue()

    proc = [mp.Process(target=_spawn(f),args=(q_in,q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]

def bin_data(data, binsize=1):
    """Bins the supplied data into of the specified size

    Args:
      data (list): The data to bin.
      binsize (int, optional): The bin width to use when resampling.

    Returns:
      list: The binned data

    Examples:
      Load some correlators and bin them.

      >>> import pyQCD
      >>> data = pyQCD.load_archive("correlators.zip")
      >>> binned_data = pyQCD.bin_data(data)
    """

    return [sum(data[i:i+binsize]) / binsize
            for i in range(0, len(data) - binsize + 1, binsize)]

def bootstrap_data(data, num_bootstraps, binsize=1, parallel=False):
    """Resamples the supplied data using the bootstrap method.
        
    Args:
      data (list): The data on which to perform a bootstrap resample.
      num_bootstraps (int): The number of bootstraps to perform.
      binsize (int, optional): The bin size to bin the data with before
        performing the bootstrap.
      parallel (bool, optional): Parallelizes the bootstrap using python's
        multiprocessing module.
             
    Returns:
      list: The resampled data set.
    
    Examples:
      Load some correlators and create bootstrap copies of them.
          
      >>> import pyQCD
      >>> data = pyQCD.load_archive("correlators.zip")
      >>> resampled_data = pyQCD.bootstrap_data(data, 100)
    """
    
    out = []

    binned_data = bin_data(data, binsize)
    num_bins = len(binned_data)

    def parallel_function(index):
        bins = npr.randint(num_bins, size=num_bins).tolist()
        return np.mean([binned_data[j] for j in bins], axis=0)

    out = _parmap(parallel_function, range(num_bootstraps)) \
      if parallel else map(parallel_function, range(num_bootstraps))

    return list(out)

def bootstrap(data, func, num_bootstraps=None, binsize=1, args=[],
              kwargs={}, resample=True, parallel=False):
    """Performs a bootstrapped measurement on the data set using the supplied
    function

    Args:
      data (list): The data on which to perform the resampling and measurement
      func (function): The measurement function. The first argument of this
        function should accept a type identical to that of the elements of the
        data list.
      num_bootstraps (int, optional): The number of bootstraps to perform. If
        resample is set to False, then this value is ignored.
      binsize (int, optional): The binsize to bin the data with before
        performing the bootstrap.
      args (list, optional): Any additional arguments required by func
      kwargs (dict, optional): Any additional keyword arguments required by func
      resample (bool, optional): Determines whether to treat data as an existing
        bootstrap data set, or perform the bootstrap from scratch.
      parallel (bool, optional): Parallelizes the bootstrap using python's
        multiprocessing module.

    Returns:
      tuple: The bootstrapped central value and the estimated error.

    Examples:
      Load some correlators and bootstrap the effective mass curve.

      >>> import pyQCD
      >>> data = pyQCD.load_archive("correlators.zip")
      >>> effmass = pyQCD.bootstrap(data, pyQCD.compute_effmass, 100)
    """

    if resample:
        resamp_data = bootstrap_data(data, num_bootstraps, binsize, parallel)
    else:
        resamp_data = data

    def parallel_function(datum):
        return func(datum, *args, **kwargs)

    results = _parmap(parallel_function, resamp_data) \
      if parallel else list(map(parallel_function, resamp_data))

    return np.mean(results, axis=0), np.std(results, axis=0)

def jackknife_data(data, binsize=1, parallel=False):
    """Resamples the supplied data using the jackknife method.
        
    Args:
      data (list): The data on which to perform a jackknife resample.
      binsize (int, optional): The bin size to bin the data with before
        performing the jackknife.
      parallel (bool, optional): Parallelizes the jackknife using python's
        multiprocessing module.
             
    Returns:
      list: The resampled data set.
    
    Examples:
      Load some correlators and create jackknife copies of the data.
          
      >>> import pyQCD
      >>> data = pyQCD.load_archive("correlators.zip")
      >>> resampled_data = pyQCD.jackknife_data(data, 100)
    """

    binned_data = bin_data(data, binsize)
    data_sum = sum(binned_data)

    def parallel_function(datum):
        return (data_sum - datum) / (len(binned_data) - 1)
    
    out = _parmap(parallel_function, binned_data) \
      if parallel else map(parallel_function, binned_data)

    return list(out)

def jackknife(data, func, binsize=1, args=[], kwargs={}, resample=True,
              parallel=False):
    """Performs a jackknifed measurement on the data set using the supplied
    function

    Args:
      data (list): The data on which to perform the resampling and measurement
      func (function): The measurement function. The first argument of this
        function should accept a type identical to that of the elements of the
        data list.
      binsize (int, optional): The binsize to bin the data with before
        performing the jackknife.
      args (list, optional): Any additional arguments required by func
      kwargs (dict, optional): Any additional keyword arguments required by func
      resample (bool, optional): Determines whether to treat data as an existing
        jackknife data set, or perform the jackknife from scratch.
      parallel (bool, optional): Parallelizes the jackknife using python's
        multiprocessing module.

    Returns:
      tuple: The jackknifed central value and the estimated error.

    Examples:
      Load some correlators and jackknife the effective mass curve.

      >>> import pyQCD
      >>> data = pyQCD.load_archive("correlators.zip")
      >>> effmass = pyQCD.jackknife(data, pyQCD.compute_effmass, 100)
    """

    if resample:
        resamp_data = jackknife_data(data, binsize, parallel)
    else:
        resamp_data = data

    def parallel_function(datum):
        return func(datum, *args, **kwargs)

    results = _parmap(parallel_function, resamp_data) \
      if parallel else list(map(parallel_function, resamp_data))

    N = len(results)

    return np.mean(results, axis=0), np.sqrt(N - 1) * np.std(results, axis=0)