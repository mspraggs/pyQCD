"""
Contains the core datatype used to store data and implement
lattice simulations and analysis.

Classes:
  * Config
  * DataSet
  * Lattice
  * Propagator
  * Simulation
  * TwoPoint
  * WilsonLoops
"""
try:
    from lattice import Lattice
    from simulation import Simulation
except ImportError:
    pass

from dicts import *    
from propagator import *
from twopoint import *
from wilslps import *
from dataset import *
from constants import *
