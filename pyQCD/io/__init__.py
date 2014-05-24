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

def _write_datum(datum, index, zfname):
    """Writes the specified file to the specified zipfile"""

    fname = "datum{}.npy".format(index)
    mode = "w" if index == 0 else "a"

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

def _extract_datum(index, zfname):
    """Extracts the specified datum from the specified zipfile"""

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

    return [_extract_datum(i, filename) for i in range(num_data)]
