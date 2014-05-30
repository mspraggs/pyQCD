"""
Contains functions to convert and save data to disk.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import zipfile

from .converters import *

def write_datum(datum, index, zfname, mode):
    """Writes the specified datum to the specified zipfile

    Args:
      datum: Structure containing numerical data. May be a scalar value, an
        iterable built-in or a numpy ndarray
      index (int): The index label to give the datum in the zip archive.
      zfname (str): The zip file name.
      mode (str): The mode in which to open the file ('w' or 'a').

    Examples:
      Create some random numpy arrays and write them to a zip archive.

      >>> import pyQCD
      >>> import numpy as np
      >>> data = [np.random.random(10) for i in range(10)]
      >>> for i, datum in enumerate(data):
      ...     pyQCD.io.write_datum(datum, i, "some_data.zip",
      ...                          'w' if i == 0 else 'a')
    """

    fname = "datum{}.npy".format(index)
    
    if zfname[-4:] != ".zip":
        zfname = "{}.zip".format(zfname)

    try:
        zfile = zipfile.ZipFile(zfname, mode, zipfile.ZIP_DEFLATED, True)
    except (RuntimeError, zipfile.LargeZipFile):
        zfile = zipfile.ZipFile(zfname, mode, zipfile.ZIP_STORED, False)

    np.save(fname, datum)
    zfile.write(fname)
    os.unlink(fname)
    zfile.close()

def extract_datum(index, zfname):
    """Extracts the specified datum from the specified zipfile

    Args:
      index (int): The index label of the required datum.
      zfname (str): The zip file name.

    Returns:
      The datum value.

    Examples:
      Load the data we saved in the pyQCD.io.write_datum example.

      >>> import pyQCD
      >>> data = [pyQCD.io.extract_datum(i, "some_data.zip")
      ...         for i in range(10)]"""

    fname = "datum{}.npy".format(index)

    if zfname[-4:] != ".zip":
        zfname = "{}.zip".format(zfname)

    try:
        zfile = zipfile.ZipFile(zfname, "r", zipfile.ZIP_DEFLATED, True)
    except (RuntimeError, zipfile.LargeZipFile):
        zfile = zipfile.ZipFile(zfname, "r", zipfile.ZIP_STORED, False)

    zfile.extract(fname)
    datum = np.load(fname)
    try:
        datum = datum.item()
    except ValueError:
        pass
    os.unlink(fname)
    zfile.close()

    return datum

def write_datum_callback(filename):
    """Creates a callback function from the write_datum function for use as a
    measurement output callback in the Simulation class.

    Args:
      filename (str): The filename to save measurments to.

    Returns:
      function: The callback function.
    """

    def wrapper(datum, index):
        write_datum(datum, index, filename, "w" if index == 0 else "a")

    return wrapper

def extract_datum_callback(filename):
    """Creates a callback function from the extract_datum function for use as an
    input loading callback in the Simulation class.

    Args:
      filename (str): The filename to load inputs from.

    Returns:
      function: The callback function.
    """

    def wrapper(index):
        datum = extract_datum(index, filename)

        return datum

    return wrapper

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
      ...         for i in range(100)]
      >>> pyQCD.save_archive("chroma_data.zip", data)
    """

    for i, datum in enumerate(data):
        write_datum(datum, i, filename, "w" if i == 0 else "a")

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

    return [extract_datum(i, filename) for i in range(num_data)]
