from observable import Observable
from propagator import Propagator
import numpy as np
import constants as const
import itertools
import scipy.optimize as spop
import re
import xml.etree.ElementTree as ET

class TwoPoint(Observable):
    """Create a two-point function from two propagators
    
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
    """
    
    common_members = ['L', 'T']
    
    def __init__(self, T, L):
        """Constructor for pyQCD.Simulation (see help(pyQCD.Simulation))"""
        self.L = L
        self.T = T
        self.computed_correlators = []
    
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
        
        for member in TwoPoint.common_members:
            header_keys.append(member)
            header_values.append(getattr(self, member))
                
        header_keys.append("computed_correlators")
        header_values.append(self.computed_correlators)

        header = dict(zip(header_keys, header_values))
        
        data_keys = []
        data_values = []
        
        for key in self.computed_correlators:
            data_keys.append(key)
            data_values.append(getattr(self, key))
            
        data = dict(zip(data_keys, data_values))
            
        np.savez(filename, header=header, **data)
        
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
        ret.computed_correlators = header['computed_correlators']
        
        for correlator in numpy_archive.keys():
            if correlator != "header":
                setattr(ret, correlator, numpy_archive[correlator])
        
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
        
        correlator_attributes \
          = [self._get_correlator_parameters(name)
             for name in self.computed_correlators]
        
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
                  
        attribute_names = [self._get_correlator_name(*attribs)
                           for attribs in correlator_attributes]
        
        if len(attribute_names) == 1:
            return getattr(self, attribute_names[0])
        else:
            attribute_values = [getattr(self, name) for name in attribute_names]
           
            return dict(zip(tuple(correlator_attributes),
                            tuple(attribute_values)))
    
    def add_correlator(self, data, label, masses=[], momentum=[0, 0, 0],
                       source_type=None, sink_type=None, projected=True):
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
        
        if projected:
            if len(data.shape) != 1 or data.shape[0] != self.T:
                raise ValueError("Expected a correlator with shape "
                                 "({},), recieved {}"
                                 .format(self.T, data.shape))
            
            setattr(self, correlator_name, data)
            
            if not correlator_name in self.computed_correlators:
                self.computed_correlators.append(correlator_name)
            
        else:
            expected_shape = (self.T, self.L, self.L, self.L)
            if data.shape != expected_shape:
                raise ValueError("Expected a correlator with shape "
                                 "{}, recieved {}"
                                 .format(expected_shape, data.shape))
            
            correlator = self._project_correlator(data, momentum)
            
            if not correlator_name in self.computed_correlators:
                self.computed_correlators.append(correlator_name)
            
            setattr(self, correlator_name, correlator)
            
    def load_chroma_mesonspec(self, filename):
        """Loads the meson correlator(s) present in the supplied Chroma
        mesonspec output xml file
        
        Args:
            filename (str): The name of the file in which the correlators
              are contained.
              
        Raises:
            ValueError: If lattice shape does not match twopoint spatial
              and temporal extents
              
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
        
        lattice_shape \
          = [int(x) for x in
             xmlroot.find("ProgramInfo/Setgeom/latt_size").text.split()]
        
        if not (lattice_shape[0] == lattice_shape[1] == lattice_shape[2]):
            raise ValueError("Chroma lattice shape has differing spatial "
                             "extents.")
        
        # Assume lattice_shape[3] is the time extent
        if lattice_shape[0] != self.L or lattice_shape[3] != self.T:
            expected_shape = (self.T, self.L, self.L, self.L)
            actual_shape = (lattice_shape[3], lattice_shape[0],
                            lattice_shape[1], lattice_shape[2])
            raise ValueError("Expected lattice shape, {}, does not match "
                             "received lattice shape, {}"
                             .format(expected_shape, actual_shape))
        
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
                                    momentum, source_type, sink_type)
            
    def load_chroma_hadspec_mesons(self, filename):
        """Loads the meson correlator(s) present in the supplied Chroma
        hadspec output xml file
        
        Args:
            filename (str): The name of the file in which the correlators
              are contained.
              
        Raises:
            ValueError: If lattice shape does not match twopoint spatial
              and temporal extents
              
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
        
        lattice_shape \
          = [int(x) for x in
             xmlroot.find("ProgramInfo/Setgeom/latt_size").text.split()]
        
        if not (lattice_shape[0] == lattice_shape[1] == lattice_shape[2]):
            raise ValueError("Chroma lattice shape has differing spatial "
                             "extents.")
        
        # Assume lattice_shape[3] is the time extent
        if lattice_shape[0] != self.L or lattice_shape[3] != self.T:
            expected_shape = (self.T, self.L, self.L, self.L)
            actual_shape = (lattice_shape[3], lattice_shape[0],
                            lattice_shape[1], lattice_shape[2])
            raise ValueError("Expected lattice shape, {}, does not match "
                             "received lattice shape, {}"
                             .format(expected_shape, actual_shape))
        
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
                                        momentum, source_type, sink_type)
            
    def load_chroma_hadspec_baryons(self, filename):
        """Loads the baryon correlator(s) present in the supplied Chroma
        hadspec output xml file
        
        Args:
            filename (str): The name of the file in which the correlators
              are contained.
              
        Raises:
            ValueError: If lattice shape does not match twopoint spatial
              and temporal extents
              
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
        
        lattice_shape \
          = [int(x) for x in
             xmlroot.find("ProgramInfo/Setgeom/latt_size").text.split()]
        
        if not (lattice_shape[0] == lattice_shape[1] == lattice_shape[2]):
            raise ValueError("Chroma lattice shape has differing spatial "
                             "extents.")
        
        # Assume lattice_shape[3] is the time extent
        if lattice_shape[0] != self.L or lattice_shape[3] != self.T:
            expected_shape = (self.T, self.L, self.L, self.L)
            actual_shape = (lattice_shape[3], lattice_shape[0],
                            lattice_shape[1], lattice_shape[2])
            raise ValueError("Expected lattice shape, {}, does not match "
                             "received lattice shape, {}"
                             .format(expected_shape, actual_shape))
        
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
                                        momentum, source_type, sink_type)
    
    def compute_meson_correlator(self, propagator1, propagator2,
                                 source_interpolator,
                                 sink_interpolator, label, momenta = [0, 0, 0],
                                 average_momenta = True):
        """Computes and stores the specified meson correlator within the
        current TwoPoint object
        
        Colour and spin traces are taken over the following product:
        
        propagator1 * source_interpolator * propagator2 * sink_interpolator
        
        :param propagator1: The first propagator to use in calculating the correlator
        :type propagator1: :class:`Propagator`
        :param propagator2: The first propagator to use in calculating the correlator
        :type propagator2: :class:`Propagator`
        :param source_interpolator: The interpolating operator describing the source of the two-point function
        :type source_interpolator: :class:`numpy.ndarray` or :class:`str`
        :param label: A label for the resulting correlator
        :type label: :class:`str`
        :param momenta: The lattice momentum (or momenta) to project the correlator onto
        :type momenta: :class:`list` of three :class:`int`s, or a compound list containing several such objects
        :param average_momenta: Determines whether equivalent momenta are averaged over
        :type average_momenta: :class:`bool`
        """
            
        if type(source_interpolator) == str:
            source_interpolator = const.Gammas[source_interpolator]
        if type(sink_interpolator) == str:
            sink_interpolator = const.Gammas[sink_interpolator]
            
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
                                momentum, source_type, sink_type)
            
    def fit_correlator(self, fit_function, fit_range, initial_parameters,
                       correlator_std=None, postprocess_function=None,
                       label=None, masses=None, momentum=None, source_type=None,
                       sink_type=None):
        """Fits the specified function to the specified correlator
        
        :param fit_function: The function to fit to the correlator
        :type fit_function: :class:`function`
        :param fit_range: two-tuple or list of integers specifying the timeslices over which to fit
        :type fit_range: :class:`tuple` or :class:`list`
        :param initial_parameters: The initial value of the fit parameters
        :type initial_parameters: :class:`tuple` or :class:`list`
        :param correlator_std: The standard deviation in the relevant correlator
        :type correlator_std: :class:`numpy.ndarray`
        :param postprocess_function: The function to apply to apply to the resulting fit parameters before returning a result
        :type postprocess_function: :class:`function`
        :param label: The correlator label
        :type label: :class:`str`
        :param masses: The correlator quark masses
        :type masses: :class:`list`
        :param momentum: The lattice momentum of the particle described by the correlator
        :type momentum: :class:`list` or :class:`tuple`
        :param source_type: The correlator source type
        :type source_type: :class:`str`
        :param sink_type: The correlator sink type
        :type sink_type: :class:`str`
        """
        
        correlator = self.get_correlator(label, masses, momentum, source_type,
                                         sink_type)
        
        if type(correlator) == dict:
            raise NameError("Correlator specifiers returned more than one "
                            "correlator.")
                
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
        
        if postprocess_function == None:
            return b
        else:
            return postprocess_function(b)
        
    def compute_energy(self, fit_range, initial_parameters, correlator_std=None,
                       label=None, masses=None, momentum=None, source_type=None,
                       sink_type=None):
        """Computes the ground state energy of the specified correlator
        
        :param fit_range: two-tuple or list of integers specifying the timeslices over which to fit
        :type fit_range: :class:`tuple` or :class:`list`
        :param initial_parameters: The initial value of the fit parameters
        :type initial_parameters: :class:`tuple` or :class:`list`
        :param correlator_std: The standard deviation in the relevant correlator
        :type correlator_std: :class:`numpy.ndarray`
        :param label: The correlator label
        :type label: :class:`str`
        :param masses: The correlator quark masses
        :type masses: :class:`list`
        :param momentum: The lattice momentum of the particle described by the correlator
        :type momentum: :class:`list` or :class:`tuple`
        :param source_type: The correlator source type
        :type source_type: :class:`str`
        :param sink_type: The correlator sink type
        :type sink_type: :class:`str`
        """
        
        fit_function \
          = lambda b, t, Ct, err: \
          (Ct - b[0] * np.exp(-b[1] * t) - b[0] * np.exp(-b[1] * (self.T - t))) \
          / err
          
        postprocess_function = lambda b: b[1]
        
        return self.fit_correlator(fit_function, fit_range, initial_parameters,
                                   correlator_std, postprocess_function, label,
                                   masses, momentum, source_type, sink_type)
        
    def compute_square_energy(self, fit_range, initial_parameters,
                              correlator_std=None, label=None, masses=None,
                              momentum=None, source_type=None, sink_type=None):
        """Computes the ground state energy of the specified correlator
        
        :param fit_range: two-tuple or list of integers specifying the timeslices over which to fit
        :type fit_range: :class:`tuple` or :class:`list`
        :param initial_parameters: The initial value of the fit parameters
        :type initial_parameters: :class:`tuple` or :class:`list`
        :param correlator_std: The standard deviation in the relevant correlator
        :type correlator_std: :class:`numpy.ndarray`
        :param label: The correlator label
        :type label: :class:`str`
        :param masses: The correlator quark masses
        :type masses: :class:`list`
        :param momentum: The lattice momentum of the particle described by the correlator
        :type momentum: :class:`list` or :class:`tuple`
        :param source_type: The correlator source type
        :type source_type: :class:`str`
        :param sink_type: The correlator sink type
        :type sink_type: :class:`str`
        """
        
        fit_function \
          = lambda b, t, Ct, err: \
          (Ct - b[0] * np.exp(-b[1] * t) - b[0] * np.exp(-b[1] * (self.T - t))) \
          / err
          
        postprocess_function = lambda b: b[1]**2
        
        return self.fit_correlator(fit_function, fit_range, initial_parameters,
                                   correlator_std, postprocess_function, label,
                                   masses, momentum, source_type, sink_type)
    
    def compute_c_square(self, fit_range, initial_parameters, momenta,
                         correlator_stds=None, label=None, masses=None,
                         source_type=None, sink_type=None):
        """Computes the square of the speed of light for the given particle
        at each of the specified momenta
        
        :param fit_range: two-tuple or list of integers specifying the timeslices over which to fit
        :type fit_range: :class:`tuple` or :class:`list`
        :param momenta: The lattice momenta or momentum at which to compute the speed of light
        :type momenta: (possibly compound) :class:`list` or :class:`tuple`
        :param initial_parameters: The initial value of the fit parameters
        :type initial_parameters: :class:`tuple` or :class:`list`
        :param correlator_stds: The standard deviation of the correlator at zero momentum and the standard deviation of the non-zero momentum correlators
        :type correlator_stds: :class:`list` of :class:`numpy.ndarray`
        :param label: The correlator label
        :type label: :class:`str`
        :param masses: The correlator quark masses
        :type masses: :class:`list`
        :param source_type: The correlator source type
        :type source_type: :class:`str`
        :param sink_type: The correlator sink type
        :type sink_type: :class:`str`
        """
        
        if type(momenta[0]) != list and type(momenta[0]) != tuple:
            momenta = [momenta]
            
        if correlator_stds == None:
            correlator_stds = (len(momenta) + 1) * [None]
        
        E0_square \
          = self.compute_square_energy(fit_range, initial_parameters,
                                       correlator_stds[0], label, masses,
                                       [0, 0, 0], source_type, sink_type)
        
        out = np.zeros(len(momenta))
        
        for i, momentum in enumerate(momenta):
            E_square = self.compute_square_energy(fit_range, initial_parameters,
                                                  correlator_stds[i + 1], label,
                                                  masses, momentum, source_type,
                                                  sink_type)
            
            p_square = sum([(2 * np.pi * x / self.L)**2 for x in momentum])
            out[i] = (E_square - E0_square) / p_square
            
        return out
    
    def compute_effmass(self, label=None, masses=None, momentum=None,
                       source_type=None, sink_type=None):
        """Computes the effective mass for the specified correlator
        
        :param label: The correlator label
        :type label: :class:`str`
        :param momentum: The lattice momentum of the correlator
        :type momentum: :class:`list`
        :param masses: The masses of the quarks in the correlator
        :type masses: :class:`list`
        :param source_type: The source type
        :type source_type: :class:`str`
        :param sink_type: The sink type
        :type sink_type: :class:`str`
        """
        
        correlator = self.get_correlator(label, masses, momentum, source_type,
                                         sink_type)
        
        if type(correlator) == dict:
            raise NameError("Correlator specifiers returned more than one "
                            "correlator.")
        
        return np.log(np.abs(correlator / np.roll(correlator, -1)))
    
    def __add__(self, tp):
        """Addition operator overload"""
        
        if type(tp) != type(self):
            raise TypeError("Types {} and {} do not match"
                            .format(type(self), type(tp)))
        
        for cm in self.common_members:
            if getattr(self, cm) != getattr(tp, cm):
                raise ValueError("Attribute {} differs between objects "
                                 "({} and {})".format(cm,
                                                      getattr(self, cm),
                                                      getattr(tp, cm)))
        
        out = TwoPoint(tp.T, tp.L)
        
        comp_corr1 = self.computed_correlators
        comp_corr2 = tp.computed_correlators
        
        for cc in comp_corr1:
            setattr(out, cc, getattr(self, cc))
            out.computed_correlators.append(cc)
            
        for cc in comp_corr2:
            if hasattr(out, cc):
                setattr(out, cc, getattr(out, cc) + getattr(tp, cc))
            else:
                setattr(out, cc, getattr(tp, cc))
                out.computed_correlators.append(cc)
                
        return out
    
    def __div__(self, div):
        """Division operator overloading"""
        
        if type(div) != int and type(div) != float:
            raise TypeError("Expected an int or float divisor, got {}"
                            .format(type(div)))
        
        out = TwoPoint(self.T, self.L)
        
        for cc in self.computed_correlators:
            setattr(out, cc, getattr(self, cc) / div)
        
        out.computed_correlators = self.computed_correlators
            
        return out
    
    def __neg__(self):
        """Negation operator overload"""
        
        out = TwoPoint(self.T, self.L)
        
        comp_corr1 = self.computed_correlators
        out.computed_correlators = comp_corr1
        
        for cc in comp_corr1:
            setattr(out, cc, -getattr(self, cc))
                
        return out
    
    def __sub__(self, tp):
        """Subtraction operator overload"""
        
        return self.__add__(tp.__neg__())
    
    def __pow__(self, exponent):
        """Power operator overloading"""
        
        if type(exponent) != int and type(exponent) != float:
            raise TypeError("Expected an int or float exponent, got {}"
                            .format(type(exponent)))
        
        out = TwoPoint(self.T, self.L)
        
        for cc in self.computed_correlators:
            setattr(out, cc, getattr(self, cc) ** exponent)
        
        out.computed_correlators = self.computed_correlators
            
        return out
                        
    def __str__(self):
        
        out = \
          "Two-Point Function Object\n" \
          "-------------------------\n" \
          "Spatial extent: {}\n" \
          "Temporal extent: {}\n\n" \
          "Computed correlators:\n" \
          "- (label, masses, momentum, source, sink)\n".format(self.L, self.T)
        
        if len(self.computed_correlators) > 0:
            for correlator in self.computed_correlators:
                correlator_parameters \
                  = self._get_correlator_parameters(correlator)
                  
                out += "- {}\n".format(correlator_parameters)
                
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
