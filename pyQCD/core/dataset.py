import os
import cPickle
import zipfile
import warnings
import multiprocessing as mp
from itertools import izip

import numpy as np
import numpy.random as npr

def _write_datum(datum, index, zfname):
    """Writes the specified file to the specified zipfile"""

    fname = "datum{}.npy".format(index)
    mode = "w" if index == 0 else "a"

    try:
        zfile = zipfile.ZipFile(zfname, mode, zipfile.ZIP_DEFLATED, True)
    except (RuntimeError, zipfile.LargeZipFile):
        zfile = zipfile.ZipFile(zfname, mode, zipfile.ZIP_STORED, False)

    np.save(fname, datum)
    zfile.write(fname)
    os.unlink(fname)
    zfile.close()

def _extract_datum(index, zfname):
    """Extracts the specified datum from the specified zipfile"""

    fname = "datum{}.npy".format(index)

    try:
        zfile = zipfile.ZipFile(zfname, "r", zipfile.ZIP_DEFLATED, True)
    except (RuntimeError, zipfile.LargeZipFile):
        zfile = zipfile.ZipFile(zfname, "r", zipfile.ZIP_STORED, False)

    zfile.extract(fname)
    datum = np.load(fname)
    os.unlink(fname)
    zfile.close()

    return datum

def save_archive(filename, data):
    """Save the supplied list of data to a zip archive using the numpy save
    function

    Args:
      filename (str): The name of the zip archive.
      data (list): The list of data to save.

    Examples:
      Load a series of correlators from CHROMA xml output files, then save
      them as a zip file.

      >>> import pyQCD
      >>> data = [pyQCD.load_chroma_hadspec("hadspec_{}.dat.xml".format(i))
      ...         for i in xrange(100)]
      >>> pyQCD.save_archive("chroma_data.zip", data)
    """

    for i, datum in enumerate(data):
        _write_datum(datum, i, filename)

def load_archive(filename):
    """Load the contents of the supplied zip archive into a list

    Args:
      filename (str): The name of the zip archive.
    """

    try:
        zfile = zipfile.ZipFile(filename, 'r', zipfile.ZIP_DEFLATED, True)
    except (RuntimeError, zipfile.LargeZipFile):
        zfile = zipfile.ZipFile(filename, 'r', zipfile.ZIP_STORED, False)

    num_data = len(zfile.filelist)

    zfile.close()

    return [_extract_datum(i, filename) for i in xrange(num_data)]

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
            for i in xrange(0, len(data) - binsize + 1, binsize)]

def bootstrap_data(data, num_bootstraps, binsize=1):
    """Resamples the supplied data using the bootstrap method.
        
    Args:
      data (list): The data on which to perform a bootstrap resample.
      num_bootstraps (int): The number of bootstraps to perform.
      binsize (int, optional): The bin size to bin the data with before
        performing the bootstrap.
             
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

    for i in xrange(num_bootstraps):
        bins = npr.randint(num_bins, size=num_bins).tolist()
        new_datum = np.mean([binned_data[j] for j in bins], axis=0)
        out.append(new_datum)

    return out

def bootstrap(data, func, num_bootstraps=None, binsize=1, args=[],
              kwargs={}, resample=True):
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

    Returns:
      tuple: The bootstrapped central value and the estimated error.

    Examples:
      Load some correlators and bootstrap the effective mass curve.

      >>> import pyQCD
      >>> data = pyQCD.load_archive("correlators.zip")
      >>> effmass = pyQCD.bootstrap(data, pyQCD.compute_effmass, 100)
    """

    if resample:
        resamp_data = bootstrap_data(data, num_bootstraps, binsize)
    else:
        resamp_data = data

    results = map(lambda x: func(x, *args, **kwargs), resamp_data)

    return np.mean(results, axis=0), np.std(results, axis=0)

def jackknife_data(data, binsize=1):
    """Resamples the supplied data using the jackknife method.
        
    Args:
      data (list): The data on which to perform a jackknife resample.
      binsize (int, optional): The bin size to bin the data with before
        performing the jackknife.
             
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

    return [(data_sum - datum) / (len(binned_data) - 1)
            for datum in binned_data]

def jackknife(data, func, binsize=1, args=[], kwargs={}, resample=True):
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

    Returns:
      tuple: The jackknifed central value and the estimated error.

    Examples:
      Load some correlators and jackknife the effective mass curve.

      >>> import pyQCD
      >>> data = pyQCD.load_archive("correlators.zip")
      >>> effmass = pyQCD.jackknife(data, pyQCD.compute_effmass, 100)
    """

    if resample:
        resamp_data = jackknife_data(data, binsize)
    else:
        resamp_data = data

    results = map(lambda x: func(x, *args, **kwargs), resamp_data)
    N = len(results)

    return np.mean(results, axis=0), np.sqrt(N - 1) * np.std(results, axis=0)
