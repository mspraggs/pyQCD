import re
import xml.etree.ElementTree as ET
import itertools
import warnings
import sys
import struct

import numpy as np
import scipy.optimize as spop

import constants as const
from observable import Observable
from propagator import Propagator

class TwoPoint(Observable):
    """Encapsulates two-point function data and provides fitting tools.
    
    The data for two-point functions is stored in member variables. Each
    individual correlator is referenced using a label, and optionally by
    the masses of the corresponding quark masses, the momentum of the
    corresponding hadron and the source and sink types used when computing
    the two-point function. These descriptors must be supplied to specify a
    unique correlator stored in the TwoPoint object when calling a function
    that operates on a correlator. For example, if several correlators are
    stored with the label "pseudoscalar", but correspond to mesons with
    different bare quark masses, then the masses can be used to distinguish
    between the correlators. Likewise, two correlators could share the same
    label but correspond to different hadron momenta.
    
    Various member functions are provided to import data from Chroma XML
    data files produced by the hadron spectrum and meson spectrum measurements.
    Data can also be imported from UKHADRON meson correlator binaries.
    Meson correlators may also be computed using pyQCD.Propagator objects.
    Correlator data may also be added by hand using the add_correlator function.
    
    Attributes:
      computed_correlators (list): A list of tuples corresponding to those
        correlators stored in the TwoPoint object. Each element in the list
        has the following format:
        [label, quark_masses, hadron_momentum, source_type, sink_type]
      L (int): The spatial extent of the lattice
      T (int): The temporal extent of the lattice
    
    Args:
      L (int): The spatial extent of the lattice
      T (int): The temporal extent of the lattice
      
    Returns:
      TwoPoint: The two-point function object.
      
    Examples:
      Create an empty TwoPoint object to hold correlators from a 16^3 x 32
      lattice.
      
      >>> import pyQCD
      >>> twopoint = pyQCD.TwoPoint(16, 32)
      
      Load a TwoPoint object from disk, then extract a correlation function
      from it labelled "pion" with quark masses (0.01, 0.01). Note that if
      this is the only correlator stored in the TwoPoint object with the
      specified quark masses, then the correlator is uniquely defined and
      it is the only correlator returned.
      
      >>> import pyQCD
      >>> twopoint = pyQCD.TwoPoint.load("correlators.npz")
      >>> twopoint.get_correlator("pion", masses=(0.01, 0.01))
      array([  9.19167425e-01,   4.41417607e-02,   4.22095090e-03,
               4.68472393e-04,   5.18833346e-05,   5.29751835e-06,
               5.84481783e-07,   6.52953123e-08,   1.59048703e-08,
               7.97830102e-08,   7.01262406e-07,   6.08545149e-06,
               5.71428481e-05,   5.05306201e-04,   4.74744759e-03,
               4.66148785e-02])
               
      Note that if there were other correlators with the same label and quark
      masses, then another descriptor would be required to specify a particular
      correlator, such as momentum. The same principle applies to functions that
      perform computations using a single correlator, such as curve fitting:
      enough descriptors must be supplied to specify a single unique correlator.
    """
    
    members = ['L', 'T']
    
    def __init__(self, T, L):
        """Constructor for pyQCD.Simulation (see help(pyQCD.Simulation))"""
        self.L = L
        self.T = T
        self.data = {}
    
    def save(self, filename):
        """Saves the two-point function to a numpy zip archive
        
        Args:
          filename (str): The name of the file in which the TwoPoint data
            will be saved.
            
        Examples:
          Create and empty TwoPoint object, add some dummy correlator data
          then save the object to disk.
          
          >>> import pyQCD
          >>> import numpy.random as npr
          >>> twopoint = pyQCD.TwoPoint(16, 32)
          >>> twopoint.add_correlator(npr.random(32), "particle_name")
          >>> twopoint.save("some_fake_correlator.npz")
        """
        
        header_keys = []
        header_values = []
        
        for member in TwoPoint.members:
            header_keys.append(member)
            header_values.append(getattr(self, member))

        header = dict(zip(header_keys, header_values))
            
        np.savez(filename, header=header, data=self.data)
        
    @classmethod
    def load(cls, filename):
        """Loads and returns a twopoint object from a numpy zip
        archive
        
        Args:
          filename (str): The filename from which to load the observable
          
        Returns:
          TwoPoint: The loaded twopoint object.
          
        Examples:
          Load a twopoint object from disk
          
          >>> import pyQCD
          >>> prop = pyQCD.TwoPoint.load("my_correlator.npz")
        """
        
        numpy_archive = np.load(filename)
        
        header = numpy_archive['header'].item()
        
        ret = TwoPoint(8, 4)
        ret.L = header['L']
        ret.T = header['T']
        ret.data = numpy_archive['data'].item()
        
        return ret
    
    def save_raw(self, filename):
        """Override the save_raw function from Observable, as the member
        variable data does not exist
        
        Args:
          filename (str): The file in which to save the data.
          
        Raises:
          NotImplementedError: Currently not implemented
        """
    
        raise NotImplementedError("TwoPoint object cannot be saved as raw "
                                  "numpy arrays")
    
    @staticmethod
    def available_interpolators():
        """Returns a list of possible interpolators for use in the
        compute_meson_correlator function
        
        Returns:
          list: Contains pairs of strings denoting interpolator names
          
          The pairs of interpolator names are equivalent. For example "pion"
          and "g5" are equivalent (g5 here denoting the fifth gamma matrix,
          which represents the creation of a pseudoscalar quark pair, of which
          the pion is one such possibility).
        """
        
        return zip(const.mesons, const.interpolators)
    
    def get_correlator(self, label=None, masses=None, momentum=None,
                       source_type=None, sink_type=None):
        """Returns the specified correlator, or a dictionary containing the
        correlators that match the arguments supplied to the function
        
        Args:
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
          >>> pion_correlators = pyQCD.TwoPoint.load("pion_correlator.npz")
          >>> pion_correlators.get_correlator("pion", momentum=(0, 0, 0))
          array([  9.19167425e-01,   4.41417607e-02,   4.22095090e-03,
                   4.68472393e-04,   5.18833346e-05,   5.29751835e-06,
                   5.84481783e-07,   6.52953123e-08,   1.59048703e-08,
                   7.97830102e-08,   7.01262406e-07,   6.08545149e-06,
                   5.71428481e-05,   5.05306201e-04,   4.74744759e-03,
                   4.66148785e-02])
        """
        
        correlator_attributes = self.data.keys()
        
        if masses != None:
            masses = tuple([round(mass, 4) for mass in masses])
            
        if momentum != None:
            momentum = tuple(momentum)
        
        filter_params = [label, masses, momentum, source_type, sink_type]
        
        for i, param in enumerate(filter_params):       
            if param != None:
                correlator_attributes \
                  = [attrib for attrib in correlator_attributes
                     if attrib[i] == param]
        
        if len(correlator_attributes) == 1:
            return self.data[correlator_attributes[0]]
        else:
            correlators = [self.data[attrib]
                           for attrib in correlator_attributes]
           
            return dict(zip(correlator_attributes,
                            tuple(correlators)))
    
    def add_correlator(self, data, label, masses=[], momentum=[0, 0, 0],
                       source_type=None, sink_type=None, projected=True,
                       fold=False):
        """Adds the supplied correlator to the current instance
        
        Args:
          data (numpy.ndarray): The correlator data. If projected is True, then
            data must have shape (T,), otherwise it should have shape
            (T, L, L, L), where T and L are the lattice temporal and spatial
            extents.
          label (str): The label for the correlator.
          masses (list, optional): The masses of the valence quarks that form
            the corresponding hadron.
          momentum (list, optional): The momentum of the corresponding hadron.
          source_type (str, optional): The type of source used when inverting
            the propagator(s) used to compute the correlator.
          sink_type (str, optional): The type of sink used when inverting the
            propagator(s) used to compute the correlator.
          projected (bool, optional): Determines whether the supplied
            correlator contains a value for every lattice site, or whether it
            has already been projected onto a fixed momentum.
          fold (bool, optional): Determines whether the correlator is folded
            about it's mid-point.
            
        Raises:
          ValueError: If the supplied correlator data does not match the
            lattice extents.
            
        Examples:
          Create an empty correlator object and add some dummy data.
          
          >>> import pyQCD
          >>> import numpy.random as npr
          >>> twopoint = pyQCD.TwoPoint(16, 32)
          >>> twopoint.add_correlator(npr.random(32), "particle_name")
        """
        correlator_name = self._get_correlator_name(label, masses, momentum,
                                                    source_type, sink_type)
        masses = tuple([round(m, 4) for m in masses])
        correlator_key = (label, masses, tuple(momentum), source_type, sink_type)
        
        if projected:
            # Reject correlators that don't match the shape that TwoPoint
            # is supposed (and save ourselves hassle later complicated
            # exceptions and so on)
            if data.shape != (self.T,):
                raise ValueError("Expected a correlator with shape "
                                 "({},), recieved {}"
                                 .format(self.T, data.shape))
            
            data = TwoPoint._fold(data) if fold else data
            self.data[correlator_key] = data
            
        else:
            # Again, save ourself the bother later
            expected_shape = (self.T, self.L, self.L, self.L)
            if data.shape != expected_shape:
                raise ValueError("Expected a correlator with shape "
                                 "{}, recieved {}"
                                 .format(expected_shape, data.shape))
            
            correlator = self._project_correlator(data, momentum)
            
            correlator = TwoPoint._fold(correlator) if fold else correlator
            self.data[correlator_key] = correlator
            
    def load_chroma_mesonspec(self, filename, fold=False):
        """Loads the meson correlator(s) present in the supplied Chroma
        mesonspec output xml file
        
        Args:
          filename (str): The name of the file in which the correlators
            are contained.
          fold (bool, optional): Determines whether the correlator is folded
            about it's mid-point.
            
        Examples:
          Create a TwoPoint object to hold correlators for a 48^3 x 96
          lattice, then load some correlators computed by Chroma's
          mesonspec routine.
          
          >>> import pyQCD
          >>> twopoint = pyQCD.TwoPoint(96, 48)
          >>> twopoint.load_chroma_mesonspec("96c48_pion_corr.xml")
        """
        
        xmlfile = ET.parse(filename)
        xmlroot = xmlfile.getroot()
        
        interpolators = xmlroot.findall("Wilson_hadron_measurements/elem")
        
        for interpolator in interpolators:
            source_particle_label = interpolator.find("source_particle").text
            sink_particle_label = interpolator.find("sink_particle").text
            
            label = "{}_{}".format(source_particle_label,
                                   sink_particle_label)
            
            mass_1 = float(interpolator.find("Mass_1").text)
            mass_2 = float(interpolator.find("Mass_2").text)
            
            raw_source_string \
              = interpolators[0] \
              .find("SourceSinkType/elem/source_type_1").text \
              .lower()
            raw_sink_string \
              = interpolators[0] \
              .find("SourceSinkType/elem/sink_type_1").text \
              .lower()
            
            source_sink_types = ["point", "shell", "wall"]
            
            for source_sink_type in source_sink_types:
                if raw_source_string.find(source_sink_type) > -1:
                    source_type = source_sink_type
                if raw_sink_string.find(source_sink_type) > -1:
                    sink_type = source_sink_type
                    
            correlator_data = interpolator.findall("Mesons/momenta/elem")
            
            for correlator_datum in correlator_data:
                
                momentum_string = correlator_datum.find("sink_mom").text
                momentum = [int(x) for x in momentum_string.split()]
                
                correlator_value_elems \
                  = correlator_datum.findall("mesprop/elem/re")
                  
                correlator = np.array([float(x.text)
                                       for x in correlator_value_elems])
                
                self.add_correlator(correlator, label, (mass_1, mass_2),
                                    momentum, source_type, sink_type, True, fold)
            
    def load_chroma_hadspec(self, filename, fold=False):
        """Loads the correlator(s) present in the supplied Chroma
        hadspec output xml file
        
        Args:
          filename (str): The name of the file in which the correlators
            are contained.
          fold (bool, optional): Determines whether the correlator is folded
            about it's mid-point.
            
        Examples:
          Create a TwoPoint object to hold correlators for a 48^3 x 96
          lattice, then load some correlators computed by Chroma's
          hadspec routine.
          
          >>> import pyQCD
          >>> twopoint = pyQCD.TwoPoint(96, 48)
          >>> twopoint.load_chroma_hadspec("96c48_hadspec_corr.xml")
        """
        
        self.load_chroma_hadspec_mesons(filename, fold)
        self.load_chroma_hadspec_baryons(filename, fold)
        self.load_chroma_hadspec_currents(filename, fold)
            
    def load_chroma_hadspec_mesons(self, filename, fold=False):
        """Loads the meson correlator(s) present in the supplied Chroma
        hadspec output xml file
        
        Args:
          filename (str): The name of the file in which the correlators
            are contained.
          fold (bool, optional): Determines whether the correlator is folded
            about it's mid-point.
            
        Examples:
          Create a TwoPoint object to hold correlators for a 48^3 x 96
          lattice, then load some correlators computed by Chroma's
          hadspec routine.
          
          >>> import pyQCD
          >>> twopoint = pyQCD.TwoPoint(96, 48)
          >>> twopoint.load_chroma_hadspec_mesons("96c48_hadspec_corr.xml")
        """
        
        xmlfile = ET.parse(filename)
        xmlroot = xmlfile.getroot()
        
        propagator_pairs = xmlroot.findall("Wilson_hadron_measurements/elem")
        
        for propagator_pair in propagator_pairs:            
            mass_1 = float(propagator_pair.find("Mass_1").text)
            mass_2 = float(propagator_pair.find("Mass_2").text)
            
            raw_source_string \
              = propagator_pairs[0] \
              .find("SourceSinkType/source_type_1").text \
              .lower()
            raw_sink_string \
              = propagator_pairs[0] \
              .find("SourceSinkType/sink_type_1").text \
              .lower()
            
            source_sink_types = ["point", "shell", "wall"]
            
            for source_sink_type in source_sink_types:
                if raw_source_string.find(source_sink_type) > -1:
                    source_type = source_sink_type
                if raw_sink_string.find(source_sink_type) > -1:
                    sink_type = source_sink_type
            
            interpolator_tag_prefix \
              = "{}_{}".format(source_type.capitalize(),
                               sink_type.capitalize())
            
            interpolators \
              = propagator_pair.findall("{}_Wilson_Mesons/elem"
                                     .format(interpolator_tag_prefix))
            
            for interpolator in interpolators:
                
                gamma_matrix \
                  = int(interpolator.find("gamma_value").text)
                label = const.mesons[gamma_matrix]
                
                correlator_data \
                  = interpolator.find("momenta")
                
                for correlator_datum in correlator_data:
                    momentum_string = correlator_datum.find("sink_mom").text
                    momentum = [int(x) for x in momentum_string.split()]
                
                    correlator_values \
                      = correlator_datum.findall("mesprop/elem/re")
                  
                    correlator = np.array([float(x.text)
                                           for x in correlator_values])
                
                    self.add_correlator(correlator, label, (mass_1, mass_2),
                                        momentum, source_type, sink_type, True,
                                        fold)
            
    def load_chroma_hadspec_baryons(self, filename, fold=False):
        """Loads the current correlator(s) present in the supplied Chroma
        hadspec output xml file
        
        Args:
          filename (str): The name of the file in which the correlators
            are contained.
          fold (bool, optional): Determines whether the correlator is folded
            about it's mid-point.
            
        Examples:
          Create a TwoPoint object to hold correlators for a 48^3 x 96
          lattice, then load some correlators computed by Chroma's
          hadspec routine.
          
          >>> import pyQCD
          >>> twopoint = pyQCD.TwoPoint(96, 48)
          >>> twopoint.load_chroma_hadspec_baryons("96c48_hadspec_corr.xml")
        """
        
        xmlfile = ET.parse(filename)
        xmlroot = xmlfile.getroot()
        
        propagator_pairs = xmlroot.findall("Wilson_hadron_measurements/elem")
        
        for propagator_pair in propagator_pairs:            
            mass_1 = float(propagator_pair.find("Mass_1").text)
            mass_2 = float(propagator_pair.find("Mass_2").text)
            
            if mass_1 == mass_2:
                baryon_names = const.baryons_degenerate
            elif mass_1 < mass_2:
                baryon_names = baryons_m1m2
            else:
                baryon_names = baryons_m2m1
            
            raw_source_string \
              = propagator_pairs[0] \
              .find("SourceSinkType/source_type_1").text \
              .lower()
            raw_sink_string \
              = propagator_pairs[0] \
              .find("SourceSinkType/sink_type_1").text \
              .lower()
            
            source_sink_types = ["point", "shell", "wall"]
            
            for source_sink_type in source_sink_types:
                if raw_source_string.find(source_sink_type) > -1:
                    source_type = source_sink_type
                if raw_sink_string.find(source_sink_type) > -1:
                    sink_type = source_sink_type
            
            interpolator_tag_prefix \
              = "{}_{}".format(source_type.capitalize(),
                               sink_type.capitalize())
            
            interpolators \
              = propagator_pair.findall("{}_Wilson_Baryons/elem"
                                     .format(interpolator_tag_prefix))
            
            for interpolator in interpolators:
                
                gamma_matrix \
                  = int(interpolator.find("baryon_num").text)
                label = baryon_names[gamma_matrix]
                
                correlator_data \
                  = interpolator.find("momenta")
                
                for correlator_datum in correlator_data:
                    momentum_string = correlator_datum.find("sink_mom").text
                    momentum = [int(x) for x in momentum_string.split()]
                
                    correlator_values \
                      = correlator_datum.findall("barprop/elem/re")
                  
                    correlator = np.array([float(x.text)
                                           for x in correlator_values])
                
                    self.add_correlator(correlator, label, (mass_1, mass_2),
                                        momentum, source_type, sink_type, True,
                                        fold)
            
    def load_chroma_hadspec_currents(self, filename, fold=False):
        """Loads the current correlator(s) present in the supplied Chroma
        hadspec output xml file
        
        Args:
          filename (str): The name of the file in which the correlators
            are contained.
          fold (bool, optional): Determines whether the correlator is folded
            about it's mid-point.
            
        Examples:
          Create a TwoPoint object to hold correlators for a 48^3 x 96
          lattice, then load some correlators computed by Chroma's
          hadspec routine.
          
          >>> import pyQCD
          >>> twopoint = pyQCD.TwoPoint(96, 48)
          >>> twopoint.load_chroma_hadspec_currents("96c48_hadspec_corr.xml")
        """
        
        xmlfile = ET.parse(filename)
        xmlroot = xmlfile.getroot()
        
        propagator_pairs = xmlroot.findall("Wilson_hadron_measurements/elem")
        
        for propagator_pair in propagator_pairs:            
            mass_1 = float(propagator_pair.find("Mass_1").text)
            mass_2 = float(propagator_pair.find("Mass_2").text)
            
            raw_source_string \
              = propagator_pairs[0] \
              .find("SourceSinkType/source_type_1").text \
              .lower()
            raw_sink_string \
              = propagator_pairs[0] \
              .find("SourceSinkType/sink_type_1").text \
              .lower()
            
            source_sink_types = ["point", "shell", "wall"]
            
            for source_sink_type in source_sink_types:
                if raw_source_string.find(source_sink_type) > -1:
                    source_type = source_sink_type
                if raw_sink_string.find(source_sink_type) > -1:
                    sink_type = source_sink_type
            
            interpolator_tag_prefix \
              = "{}_{}".format(source_type.capitalize(),
                               sink_type.capitalize())
            
            vector_currents \
              = propagator_pair.findall("{}_Meson_Currents/Vector_currents/elem"
                                     .format(interpolator_tag_prefix))
            
            for vector_current in vector_currents:
                
                current_num \
                  = int(vector_current.find("current_value").text)
                label = const.vector_currents[current_num]
                
                correlator_data = vector_current.find("vector_current").text
                
                correlator \
                  = np.array([float(x) for x in correlator_data.split()])
                
                self.add_correlator(correlator, label, (mass_1, mass_2),
                                    [0, 0, 0], source_type, sink_type, True,
                                    fold)
            
            axial_currents \
              = propagator_pair.findall("{}_Meson_Currents/Axial_currents/elem"
                                     .format(interpolator_tag_prefix))
            
            for axial_current in axial_currents:
                
                current_num \
                  = int(axial_current.find("current_value").text)
                label = const.axial_currents[current_num]
                
                correlator_data = axial_current.find("axial_current").text
                
                correlator \
                  = np.array([float(x) for x in correlator_data.split()])
                
                self.add_correlator(correlator, label, (mass_1, mass_2),
                                    [0, 0, 0], source_type, sink_type, True,
                                    fold)
                
    def load_ukhadron_meson_binary(self, filename, byteorder, quark_masses,
                                   source_type=None, sink_type=None,
                                   fold=False):
        """Loads the correlators contained in the specified UKHADRON binary
        file. The correlators are labelled using the CHROMA convention for
        particle interpolators (see pyQCD.mesons). Note that information
        on the quark masses and the source and sink types is extracted
        from the filename, so if this information is missing then the function
        will fail.
        
        Args:
            filename (str): The name of the file containing the data
            byteorder (str): The endianness of the binary file. Can either be
              "little" or "big". For data created from simulations on an intel
              machine, this is likely to be "little". For data created on one
              of the IBM Bluegene machines, this is likely to be "big".
            quark_masses (list): The masses of the quarks forming the meson that
              the correlators represent.
            source_type (str, optional): The type of source used when computing
              the two-point function.
            sink_type (str, optional): The type of sink used when computing
              the two-point function.
            fold (bool, optional): Determines whether the correlators should
              be folded prior to being imported.
              
        Examples:
          Create a twopoint object and import the data contained in
          meson_m_0.45_m_0.45_Z2.280.bin
          
          >>> import pyQCD
          >>> twopoint = pyQCD.TwoPoint(L=16, T=32)
          >>> twopoint \
          ...     .load_ukhadron_meson_binary("meson_m_0.45_m_0.45_Z2.280.bin",
          ...                                 "big")
        """
        
        if sys.byteorder == byteorder:
            switch_endianness = lambda x: x
        else:
            switch_endianness = lambda x: x[::-1]
        
        binary_file = open(filename)
        
        mom_num_string = switch_endianness(binary_file.read(4))
        mom_num = struct.unpack('i', mom_num_string)[0]

        for i in xrange(mom_num):
            header_string = binary_file.read(40)
            px = struct.unpack('i', switch_endianness(header_string[16:20]))[0]
            py = struct.unpack('i', switch_endianness(header_string[20:24]))[0]
            pz = struct.unpack('i', switch_endianness(header_string[24:28]))[0]
            Nmu = struct.unpack('i', switch_endianness(header_string[28:32]))[0]
            Nnu = struct.unpack('i', switch_endianness(header_string[32:36]))[0]
            T = struct.unpack('i', switch_endianness(header_string[36:40]))[0]
    
            correlators = np.zeros((Nmu, Nnu, T))
    
            for t in xrange(T):
                for mu in xrange(Nmu):
                    for nu in xrange(Nnu):
                        double_string = switch_endianness(binary_file.read(8))
                        correlators[mu, nu, t] \
                          = struct.unpack('d', double_string)[0]
                        double_string = binary_file.read(8)
                        
            for mu in xrange(Nmu):
                for nu in xrange(Nnu):
                    label = "{}_{}".format(const.interpolators[mu],
                                           const.interpolators[nu])
                    self.add_correlator(correlators[mu, nu], label, quark_masses,
                                        [px, py, pz], source_type, sink_type,
                                        True, fold)
    
    def compute_meson_correlator(self, propagator1, propagator2,
                                 source_interpolator,
                                 sink_interpolator, label, momenta = [0, 0, 0],
                                 average_momenta=True,
                                 fold=False):
        """Computes and stores the specified meson correlator within the
        current TwoPoint object
        
        Colour and spin traces are taken over the following product:
        
        propagator1 * source_interpolator * propagator2 * sink_interpolator
        
        Args:
          propagator1 (Propagator): The first propagator to use in calculating
            the correlator.
          propagator2 (Propagator): The second propagator to use in calculating
            the correlator.
          source_interpolator (numpy.ndarray or str): The interpolator
            describing the source of the two-point function. If a numpy array
            is passed, then it must have the shape (4, 4) (i.e. must encode
            some form of spin structure). If a string is passed, then the
            operator is searched for in pyQCD.constants.Gammas. A list of
            possible strings to use as this argument can be seen by calling
            TwoPoint.available_interpolators()
          sink_interpolator (numpy.ndarray or str): The interpolator describing
            the sink of the two-point function. Conventions for this argument
            follow those of source_interpolator.
          label (str): The label ascribed to the resulting correlator.
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
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> prop = lattice.get_propagator(0.4)
          >>> twopoint = pyQCD.TwoPoint(8, 4)
          >>> twopoint.compute_meson_correlator(prop, prop, "g5", "g5"
          ...                                   "pseudoscalar")
        """
        
        try:
            source_interpolator = const.Gammas[source_interpolator]
        except TypeError:
            pass
        
        try:
            sink_interpolator = const.Gammas[sink_interpolator]
        except TypeError:
            pass
            
        if type(momenta[0]) != list and type(momenta[0]) != tuple:
            momenta = [momenta]
        
        spatial_correlator = self._compute_correlator(propagator1, propagator2,
                                                      source_interpolator,
                                                      sink_interpolator)
        
        if propagator1.num_source_smears == 0 \
          and propagator2.num_source_smears == 0:
            source_type = "point"
        elif propagator1.num_source_smears > 0 \
          and propagator2.num_source_smears > 0:
            source_type = "shell"
        else:
            source_type = None
            
        if propagator1.num_sink_smears == 0 \
          and propagator2.num_sink_smears == 0:
            sink_type = "point"
        elif propagator1.num_sink_smears > 0 \
          and propagator2.num_sink_smears > 0:
            sink_type = "shell"
        else:
            sink_type = None
        
        # Now go through all momenta and compute the
        # correlators
        for momentum in momenta:
            if average_momenta:
                equiv_momenta = self._get_all_momenta(momentum)
                equiv_correlators = np.zeros((len(equiv_momenta), self.T))
                
                for i, equiv_momentum in enumerate(equiv_momenta):
                    equiv_correlators[i] \
                      = self._project_correlator(spatial_correlator,
                                                 equiv_momentum)
                    
                correlator = np.mean(equiv_correlators, axis=0)
                
            else:
                correlator = self._project_correlator(spatial_correlator,
                                                      momentum)
            
            self.add_correlator(correlator, label,
                                [propagator1.mass, propagator2.mass],
                                momentum, source_type, sink_type, True, fold)
    
    def compute_all_meson_correlators(self, propagator1, propagator2,
                                      momenta = [0, 0, 0],
                                      average_momenta=True,
                                      fold=False):
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
          >>> lattice = pyQCD.Lattice()
          >>> lattice.thermalize(100)
          >>> prop = lattice.get_propagator(0.4)
          >>> twopoint = pyQCD.TwoPoint(8, 4)
          >>> twopoint.compute_all_meson_correlators(prop, prop)
        """
        
        for Gamma1 in const.interpolators:
            for Gamma2 in const.interpolators:
                self.compute_meson_correlator(propagator1, propagator2,
                                              Gamma1, Gamma2,
                                              "{}_{}".format(Gamma1, Gamma2),
                                              momenta, average_momenta,
                                              fold)
            
    def fit_correlator(self, fit_function, fit_range, initial_parameters,
                       correlator_std=None, postprocess_function=None,
                       label=None, masses=None, momentum=None, source_type=None,
                       sink_type=None):
        """Fits the specified function to the specified correlator using
        scipy.optimize.leastsq
                     
        Args:
          fit_function (function): The function with which to fit the
            correlator. Must accept a list of fitting parameters as
            the first argument, followed by a numpy.ndarray of time
            coordinates, a numpy.ndarray of correlator values and a
            numpy.ndarray of correlator errors.
          fit_range (list or tuple): Specifies the timeslices over which
            to perform the fit. If a list or tuple with two elements is
            supplied, then range(*fit_range): is applied to the function
            to generate a list of timeslices to fit over.
          initial_parameters (list or tuple): The initial parameters to
            supply to the fitting routine.
          correlator_std (numpy.ndarray, optional): The standard deviation
            in the specified correlator. If no standard deviation is
            supplied, then it is taken to be unity for each timeslice.
            This is equivalent to neglecting the error when computing
            the residuals for the fit.
          postprocess_function (function, optional): The function to apply
            to the result from scipy.optimize.leastsq.
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
          list: The fitted parameters for the fit function.
            
        Examples:
          Load a correlator from disk and fit a simple exponential to it.
          Because there is only one correlator in the loaded TwoPoint
          object, no specifiers need to be provided, since get_correlator
          will return a unique correlator regardless. A postprocess
          function to select the mass from the fit result is also
          specified.
          
          >>> import pyQCD
          >>> import numpy as np
          >>> correlator = pyQCD.TwoPoint.load("my_correlator.npz")
          >>> def fit_function(b, t, Ct, err):
          ...     return (Ct - b[0] * np.exp(-b[1] * t)) / err
          ...
          >>> postprocess = lambda b: b[1]
          >>> correlator.fit_correlator(fit_function, [5, 10], [1., 1.],
          ...                           postprocess_function=postprocess)
          1.356389
          
          Load a TwoPoint DataSet from disk and jackknife an exponential
          fit. We use DataSet.statistics to get an estimate of the error
          in the correlator.
          
          >>> import pyQCD
          >>> import numpy as np
          >>> correlator_data = pyQCD.DataSet.load("correlators.zip")
          >>> def fit_function(b, t, Ct, err):
          ...     return (Ct - b[0] * np.exp(-b[1] * t)) / err
          ...
          >>> postprocess = lambda b: b[1]
          >>> stats = correlator_data.statistics()
          >>> correlator_std = stats[1].get_correlator("pion")
          >>> correlator_data.jackknife(pyQCD.TwoPoint.fit_correlator,
          ...                           args=[fit_function, [5, 10],
          ...                                 [1., 1.], correlator_std,
          ...                                 postprocess, "pion"])
          (1.356541, 0.088433)
        """
        
        correlator = self.get_correlator(label, masses, momentum, source_type,
                                         sink_type)
                
        if correlator_std == None:
            correlator_std = np.ones(self.T)
        if len(fit_range) == 2:
            fit_range = range(*fit_range)
            
        t = np.arange(self.T)
        
        x = t[fit_range]
        y = correlator[fit_range]
        err = correlator_std[fit_range]
        
        b, result = spop.leastsq(fit_function, initial_parameters,
                                 args=(x, y, err))
        
        if [1, 2, 3, 4].count(result) < 1:
            warnings.warn("fit failed when calculating potential at "
                          "r = {}".format(r), RuntimeWarning)
        
        if postprocess_function == None:
            return b
        else:
            return postprocess_function(b)
        
    def compute_energy(self, fit_range, initial_parameters, correlator_std=None,
                       label=None, masses=None, momentum=None, source_type=None,
                       sink_type=None):
        """Computes the ground state energy of the specified correlator
        by fitting a curve to the data. The type of curve to be fitted
        (cosh or sinh) is determined from the shape of the correlator.
                     
        Args:
          fit_range (list): (list or tuple): Specifies the timeslices over which
            to perform the fit. If a list or tuple with two elements is
            supplied, then range(*fit_range): is applied to the function
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
          float: The fitted ground state energy.
          
        Examples:
          This function works in a very similar way to fit_correlator
          function, except the fitting function and the postprocessing
          function are already specified.
          
          >>> import pyQCD
          >>> correlator = pyQCD.TwoPoint.load("correlator.npz")
          >>> correlator.compute_energy([5, 16], [1.0, 1.0])
          1.532435
        """
        
        correlator = self.get_correlator(label, masses, momentum, source_type,
                                         sink_type)
        
        if TwoPoint._detect_cosh(correlator):
            fit_function \
              = lambda b, t, Ct, err: \
              (Ct - b[0] * np.exp(-b[1] * t)
               - b[0] * np.exp(-b[1] * (self.T - t))) \
              / err
        else:
            fit_function \
              = lambda b, t, Ct, err: \
              (Ct - b[0] * np.exp(-b[1] * t)
               + b[0] * np.exp(-b[1] * (self.T - t))) \
              / err
          
        postprocess_function = lambda b: b[1]
        
        return self.fit_correlator(fit_function, fit_range, initial_parameters,
                                   correlator_std, postprocess_function, label,
                                   masses, momentum, source_type, sink_type)
        
    def compute_square_energy(self, fit_range, initial_parameters,
                              correlator_std=None, label=None, masses=None,
                              momentum=None, source_type=None, sink_type=None):
        """Computes the square of the ground state energy of the specified
        correlator by fitting a curve to the data. The type of curve to be
        fitted (cosh or sinh) is determined from the shape of the correlator.
                     
        Args:
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
        
        return self.compute_energy(fit_range, initial_parameters, correlator_std,
                                   label, masses, momentum, source_type,
                                   sink_type) ** 2
    
    def compute_c_square(self, fit_ranges, initial_parameters, momenta,
                         correlator_stds=None, label=None, masses=None,
                         source_type=None, sink_type=None):
        """Computes the square of the speed of light for the given particle
        at each of the specified momenta. This calculation is performed by
        computing the square of the ground state energy for each of the
        non-zero momentum correlators. The square of the ground state energy
        for the zero-momentum correlator is then subtracted from these values,
        before each of these differences is divided by the lattice momenta,
        equal to 2 * pi * p / L.
        
        Args:
          fit_ranges (list): A compound list containing the timeslices
            over which to fit the correlator. These are specified in the
            same way as in compute_energy or compute_square_energy.
          initial_parameters (list): The initial parameters to use in
            performing.
          correlator_stds (list, optional): The standard deviations in the
            corresponding correlators as numpy.ndarrays.
          label (str, optional): The correlator label.
          masses (list, optional): The masses of the quarks forming the
            particle being studied.
          source_type (str, optional): The type of source used when
            generating the propagators that form the correlator.
          sink_type (str, optional): The type of sink used when
            generating the propagators that form the correlator.
            
        Returns:
          np.ndarray: The speeds of light squared
          
          The positions of the values in this array correspond directly
          to the positions in the momenta variable.
            
        Examples:
          This function works in a similar way to the compute_energy
          and compute_square_energy functions. Here we load a set of
          correlators and compute the speed of light squared at the
          first three non-zero lattice momenta.
          
          >>> import pyQCD
          >>> correlators = pyQCD.TwoPoint("correlators.npz")
          >>> correlators.compute_c_square([[4, 10], [6, 12],
          ...                               [7, 13], [8, 13]],
          ...                              [1.0, 1.0],
          ...                              [[1, 0, 0], [1, 1, 0]
          ...                               [1, 1, 1]])
          array([ 0.983245,  0.952324, 0.928973])
        """
        
        if type(momenta[0]) != list and type(momenta[0]) != tuple:
            momenta = [momenta]
            
        if correlator_stds == None:
            correlator_stds = (len(momenta) + 1) * [None]
        
        E0_square \
          = self.compute_square_energy(fit_ranges[0], initial_parameters,
                                       correlator_stds[0], label, masses,
                                       [0, 0, 0], source_type, sink_type)
        
        out = np.zeros(len(momenta))
        
        for i, momentum in enumerate(momenta):
            E_square = self.compute_square_energy(fit_ranges[i + 1],
                                                  initial_parameters,
                                                  correlator_stds[i + 1], label,
                                                  masses, momentum, source_type,
                                                  sink_type)
            
            p_square = sum([(2 * np.pi * x / self.L)**2 for x in momentum])
            out[i] = (E_square - E0_square) / p_square
            
        return out
    
    def compute_effmass(self, label=None, masses=None, momentum=None,
                       source_type=None, sink_type=None):
        """Computes the effective mass for the specified correlator
        by computing log(C(t) / C(t + 1))
        
        Args:
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
          numpy.ndarray: The effective mass.
            
        Examples:
          Load a TwoPoint object containing a single correlator and
          compute its effective mass.
          
          >>> import pyQCD
          >>> correlator = pyQCD.TwoPoint.load("mycorrelator.npz")
          >>> correlator.compute_effmass()
          array([ 0.44806453,  0.41769303,  0.38761196,  0.3540968 ,
                  0.3112345 ,  0.2511803 ,  0.16695767,  0.05906789,
                 -0.05906789, -0.16695767, -0.2511803 , -0.3112345 ,
                 -0.3540968 , -0.38761196, -0.41769303, -0.44806453])
        """
        
        correlator = self.get_correlator(label, masses, momentum, source_type,
                                         sink_type)
        
        return np.log(np.abs(correlator / np.roll(correlator, -1)))
    
    def __add__(self, tp):
        """Addition operator overload"""
        
        out = TwoPoint(tp.T, tp.L)
        
        for key in self.data.keys():
            out.data[key] = self.data[key] + tp.data[key]
                
        return out
    
    def __div__(self, div):
        """Division operator overloading"""
        
        out = TwoPoint(self.T, self.L)
        
        new_correlators = [correlator / div
                           for correlator in self.data.values()]
            
        out.data = dict(zip(self.data.keys(), new_correlators))
            
        return out
    
    def __neg__(self):
        """Negation operator overload"""
        
        out = TwoPoint(self.T, self.L)
        
        new_correlators = [-correlator
                           for correlator in self.data.values()]
            
        out.data = dict(zip(self.data.keys(), new_correlators))
                
        return out
    
    def __sub__(self, tp):
        """Subtraction operator overload"""
        
        return self.__add__(tp.__neg__())
    
    def __pow__(self, exponent):
        """Power operator overloading"""
        
        out = TwoPoint(self.T, self.L)
        
        new_correlators = [correlator ** exponent
                           for correlator in self.data.values()]
            
        out.data = dict(zip(self.data.keys(), new_correlators))
        
        out.computed_correlators = self.data
            
        return out
                        
    def __str__(self):
        
        out = \
          "Two-Point Function Object\n" \
          "-------------------------\n" \
          "Spatial extent: {}\n" \
          "Temporal extent: {}\n\n" \
          "Computed correlators:\n" \
          "- (label, masses, momentum, source, sink)\n".format(self.L, self.T)
        
        if len(self.data.keys()) > 0:
            for correlator in self.data.keys():                  
                out += "- {}\n".format(correlator)
                
        else:
            out += "None\n"
        
        return out

    @staticmethod
    def _compute_correlator(prop1, prop2, gamma1, gamma2):
        """Calculates the correlator for all space-time points
        
        We're doing the following (g1 = gamma1, g2 = gamma2, p1 = self.prop1,
        p2 = self.prop2, g5 = const.gamma5):
        
        sum_{spin,colour} g1 * g5 * p1 * g5 * g2 * p2
        
        The g5s are used to find the Hermitian conjugate of the first
        propagator
        """
        
        gp1 = np.matrix(gamma1) * prop1.adjoint()
        gp2 = np.matrix(gamma2) * prop2
        
        return np.einsum('txyzijab,txyzjiba->txyz',
                         gp1.data, gp2.data).real
    
    def _project_correlator(self, spatial_correlator, momentum):
        """Projects the supplied spatial correlator onto a given momentum"""
            
        sites = list(itertools.product(xrange(self.L),
                                       xrange(self.L),
                                       xrange(self.L)))
        exponential_prefactors \
          = np.exp(2 * np.pi / self.L * 1j * np.dot(sites, momentum))
            
        correlator = np.dot(np.reshape(spatial_correlator, (self.T, self.L**3)),
                            exponential_prefactors).real
        return correlator
    
    @staticmethod
    def _get_correlator_name(label, quark_masses, lattice_momentum,
                             source_type, sink_type):
        """Generates the member name of the correlator"""
        
        momentum_string = "_px{0}_py{1}_pz{2}".format(*lattice_momentum)
        mass_string = "".join(["_M{0}".format(round(mass, 4)).replace(".", "p")
                               for mass in quark_masses])
        source_sink_string = "_{0}_{1}".format(source_type, sink_type)
        
        return "{0}{1}{2}{3}".format(label, momentum_string, mass_string,
                                     source_sink_string)
    
    @staticmethod
    def _get_correlator_parameters(attribute_name):
        """Parses the attribute name and returns the parameters in the
        attribute name"""
        
        attribute_mask \
          = r'(\w+)_px(\d+)_py(\d+)_pz(\d+)(_\w+)_([a-zA-Z]+)_([a-zA-Z]+)'
        
        split_attribute_name = re.findall(attribute_mask, attribute_name)
        
        if len(split_attribute_name) == 0:
            raise ValueError("Unable to parse correlator name: {0}"
                             .format(attribute_name))
        
        split_attribute_name = split_attribute_name[0]
                
        label = split_attribute_name[0]
        mass_attributes = re.findall(r'M(\d+p\d+)', split_attribute_name[4])
        
        momentum = [eval(p) for p in split_attribute_name[1:4]]
        masses = [eval(mass.replace("p", ".")) for mass in mass_attributes]
        source_type = split_attribute_name[5]
        sink_type = split_attribute_name[6]
        
        return label, tuple(masses), tuple(momentum), source_type, sink_type

    @staticmethod
    def _get_all_momenta(p):
        """Generates all possible equivalent lattice momenta
        
        :param p: The lattice momentum to find equivalent momenta of
        :type p: :class:`list` with three elements
        :returns: :class:`list` containing the equivalent momenta
        """
        temp_list = []
        out = []
        p_rev = p[::-1]
        
        for i in xrange(3):
            temp_list.append([p[(j + i) % 3] for j in xrange(3)])
            temp_list.append([-p[(j + i) % 3] for j in xrange(3)])
            temp_list.append([p_rev[(j + i) % 3] for j in xrange(3)])
            temp_list.append([-p_rev[(j + i) % 3] for j in xrange(3)])
            
        # Remove duplicates
        for p2 in temp_list:
            if not p2 in out:
                out.append(p2)
            
        out.sort()
            
        return out
    
    @staticmethod
    def _chi_squared(b, t, Ct, err, fit_function, b_est=None, b_est_err=None):
        """Computes the chi squared value for the supplied
        data, fit function and fit parameters"""
        
        residuals = (Ct - fit_function(b, t)) / err
        
        if b_est != None and b_est_err != None:
            b = np.array(b)
            b_est = np.array(b_est)
            b_est_err = np.array(b_est_err)
            
            param_residuals = (b - b_est) / b_est_err
        else:
            param_residuals = 0
            
        return np.sum(residuals**2) + np.sum(param_residuals**2)
    
    @staticmethod
    def _detect_cosh(x):
        if np.sign(x[1]) == np.sign(x[-1]):
            return True
        else:
            return False
    
    @staticmethod
    def _fold_cosh(x):
        return np.append(x[0], (x[:0:-1] + x[1:]) / 2)
    
    @staticmethod
    def _fold_sinh(x):
        return np.append(x[0], (x[1:] - x[:0:-1]) / 2)
    
    @staticmethod
    def _fold(x):
        if TwoPoint._detect_cosh(x):
            return TwoPoint._fold_cosh(x)
        else:
            return TwoPoint._fold_sinh(x)
