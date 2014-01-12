from observable import Observable
from propagator import Propagator
import numpy as np
import constants as const
import itertools
import scipy.optimize as spop
import re

class TwoPoint(Observable):
    
    common_members = ['L', 'T']
    
    def __init__(self, T, L):
        """Create a two-point function from two propagators
        
        :param L: The spatial extent of the lattice
        :type L: :class:`int`
        :param T: The second propagator in the two-point function
        :type T: :class:`int`
        :raises: ValueError
        """
        self.L = L
        self.T = T
        self.computed_correlators = []
    
    def save(self, filename):
        """Saves the two-point function to a numpy zip archive
        
        :param filename: The file to save to
        :type filename: :class`str`
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
        """Loads and returns a two-point function from a numpy zip
        archive
        
        :param filename: The file to load from
        :type filename: :class:`str`
        :returns: :class:`TwoPoint`
        """
        
        numpy_archive = np.load(filename)
        
        header = numpy_archive['header'].item()
        
        ret = TwoPoint(8, 4)
        ret.L = header['L']
        ret.T = header['T']
        ret.comuted_correlators = header['computed_correlators']
        
        for correlator in numpy_archive.keys():
            if ['header'].count(correlator) == 0:
                setattr(ret, correlator, numpy_archive[correlator])
        
        return ret
    
    def save_raw(self, filename):
        """Override the save_raw function from Observable, as the member
        variable data does not exist
        
        :raises: NotImplementedError
        """
    
        raise NotImplementedError("TwoPoint object cannot be saved as raw "
                                  "numpy arrays")
    
    @staticmethod
    def available_interpolators():
        """Returns a list of possible interpolators for use in the
        meson_correlator function
        
        :returns: :class:`list` of tuples, each describing the meson state and the gamma matrix combination associated with it
        """
        
        return zip(const.mesons, const.interpolators)
    
    def get_correlator(self, label=None, momentum=None, masses=None,
                       source_type=None, sink_type=None):
        """Returns the specified correlator, or a dictionary containing the
        correlators that match the arguments supplied to the function
        
        :param label: The correlator label
        :type label: :class:`str`
        :param momentum: The lattice momentum of the correlator
        :type momentum: :class:`list`
        :param masses: The masses of the quarks in the correlator
        :type masses: :class:`list`
        :param source_label: The source type
        :type source_label: :class:`str`
        :param sink_label: The sink type
        :type sink_label: :class:`str`
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
        
        :param data: The correlator itself
        :type data: 1D :class:`numpy.ndarray` if projected is True, otherwise 4D :class:`numpy.ndarray` with shape (T, L, L, L)
        :param label: The label for the correlator
        :type label: :class:`str`
        :param masses: The masses of the quarks forming the particle
        :type masses: :class:`list`
        :param momentum: The momentum of the particle
        :type momentum: :class:`list` with three integers
        :param source_type: The nature of the source (smeared, point, etc...)
        :type source_type: :class:`str`
        :param sink_type: The nature of the sink (smeared, point, etc...)
        :type source_type: :class:`str`
        :param projected: Whether the correlator has been projected onto the specified momentum
        :type projected: :class:`bool`
        :raises: ValueError
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
                
    def compute_energy(self, particles, fit_range, momenta = [0, 0, 0],
                       average_momenta = True, stddev = None,
                       return_amplitude=False, fit_function=None):
        """Computes the energy of the specified particles at the specified
        momenta
        
        :param particles: The particle or list of particle to find the energy of
        :type particles: :class:`str` or :class:`list`
        :param fit_range: The two time slices specifying the range to fit over
        :type fit_range: :class:`list` of two :class:`int`s
        :param momenta: The momentum or a list of momenta to compute the energy for
        :type momenta: :class:`list`
        :param average_momenta: Determines whether equivalent momenta should be averaged over
        :type average_momenta: :class:`bool`
        :param stddev: The standard deviation in the correlators of the specified particles and momenta
        :type stddev: :class:`dict` with keys of the form (particle, momentum) for each correlator, or a single :class:`numpy.ndarray` where one particle and momentum are specified
        :param return_amplitude: Determines whether the square amplitude is returned
        :type return_amplitude: :class:`bool`
        :param fit_function: The function used to fit to. Defaults to a cosh-like function
        :type fit_function: :class:`function`
        :returns: :class:`dict` with keys specifying the particles and momenta, along with the corresponding energies and, where applicable, the square amplitudes
        """
        
        # First make sure that all relevant correlators have been calculated
        if type(momenta[0]) != list:
            momenta = [momenta]
            
        if type(particles) != list:
            particles = [particles]
        
        num_correlators = len(particles) * len(momenta)
            
        if stddev == None:
            naive_stddevs = [np.ones(self.T) for i in xrange(num_correlators)]
            tuple_momenta = [tuple(p) for p in momenta]
            keys = itertools.product(particles, tuple_momenta)
            pairs = tuple(zip(keys, naive_stddevs))
            stddev = dict(pairs)
        
        if num_correlators == 1 and type(stddev) != dict:
            stddev = {(particles[0], tuple(momenta[0])) : stddev}
            
        if len(stddev.keys()) < num_correlators:
            raise ValueError("Number of supplied correlator standard deviations "
                             "({}) does not match the specified number of "
                             "correlators ({})".format(len(stddev.keys()),
                                                       num_correlators))
            
        outstanding_particles = []
        outstanding_momenta = []
            
        for particle in particles:
            for momentum in momenta:
                if not hasattr(self, "{}_px{}_py{}_pz{}".format(particle,
                                                                *momentum)):
                    outstanding_particles.append(particle)
                    outstanding_momenta.append(momentum)
                    
        if len(outstanding_particles) > 0 and len(outstanding_momenta):
            self.meson_correlator(outstanding_particles, outstanding_momenta,
                                  average_momenta)
        
        energies = []
        keys = []
        
        # Specify the fit function
        if fit_function == None:
            fit_function = lambda b, t: \
              b[0] * (np.exp(-b[1] * (self.T - t)) + np.exp(-b[1] * t))

        for particle in particles:
            for momentum in momenta:
                attrib_name = "{}_px{}_py{}_pz{}".format(particle,
                                                         *momentum)
                current_correlator \
                  = getattr(self, attrib_name)
                x = np.arange(current_correlator.size)[fit_range[0]:
                                                       fit_range[1]]
                y = current_correlator[fit_range[0]:fit_range[1]]
                yerr = stddev[(particle, tuple(momentum))][fit_range[0]:
                                                           fit_range[1]]
                  
                result = spop.minimize(TwoPoint._chi_squared,
                                       [current_correlator[0], 1.0],
                                       args=(x, y, yerr, fit_function),
                                       method="Powell")
                
                if not result['success']:
                    print("Warning: fit failed for {} with momentum {}"
                          .format(particle, momentum))
                
                result_values = result['x']
                result_values[0] *= (2 * result_values[1])
                if return_amplitude:
                    energies.append(result_values)
                else:
                    energies.append(result_values[1])
                keys.append(attrib_name)
                
        return dict(zip(keys, energies))
                
    def compute_square_energy(self, particles, fit_range, momenta = [0, 0, 0],
                              average_momenta = True, stddev=None):
        """Computes the square energy of the specified particles at the specified
        momenta
        
        :param particles: The particle or list of particle to find the energy of
        :type particles: :class:`str` or :class:`list`
        :param fit_range: The two time slices specifying the range to fit over
        :type fit_range: :class:`list` of two :class:`int`s
        :param momenta: The momentum or a list of momenta to compute the energy for
        :type momenta: :class:`list`
        :param average_momenta: Determines whether equivalent momenta should be averaged over
        :type average_momenta: :class:`bool`
        :param stddev: The standard deviation in the correlators of the specified particles and momenta
        :type stddev: :class:`dict` with keys of the form (particle, momentum) for each correlator, or a single :class:`numpy.ndarray` where one particle and momentum are specified
        :returns: :class:`dict` with keys specifying particles and momenta
        """
        
        energies = self.compute_energy(particles, fit_range, momenta,
                                       average_momenta, stddev)
        
        return dict(zip(energies.keys(),
                        [x**2 for x in energies.values()]))
    
    def compute_c_square(self, particle, fit_range, momenta,
                         average_momentum=True, use_lattice_momenta=True,
                         stddev=None):
        """Computes the square speed of light of the specified particles at the
        specified momenta
        
        :param particle: The particle to find the speed of light of
        :type particles: :class:`str`
        :param fit_range: The two time slices specifying the range to fit over
        :type fit_range: :class:`list` of two :class:`int`s
        :param momenta: The list of momenta to compute the speed of light for
        :type momenta: :class:`list`
        :param average_momenta: Determines whether equivalent momenta should be averaged over
        :type average_momenta: :class:`bool`
        :param use_lattice_momenta: Determines whether a sine function is used when implementing the dispersion relation
        :type use_lattice_momenta: :class: `bool`
        :param stddev: The standard deviation in the correlators of the specified particles and momenta
        :type stddev: :class:`dict` with keys of the form (particle, momentum) for each correlator, or a single :class:`numpy.ndarray` where one particle and momentum are specified
        :returns: :class:`list` with entries corresponding to the supplied momenta
        """
        
        if type(momenta[0]) != list:
            momenta = [momenta]
        E0_square = self.compute_square_energy(particle, fit_range, [0, 0, 0],
                                               stddev)
        Es_square = self.compute_square_energy(particle, fit_range, momenta,
                                               stddev)
        
        out = []
        
        for p in momenta:
            E_square = Es_square["{}_px{}_py{}_pz{}".format(particle, *p)]
            
            if use_lattice_momenta:
                p_square = sum([np.sin(2 * np.pi * x / self.L)**2 for x in p])
            else:
                p_square = sum([(2 * np.pi * x / self.L)**2 for x in p])
            
            c_square = (E_square - E0_square["{}_px0_py0_pz0".format(particle)]) \
              / p_square
              
            out.append(c_square)
            
        return out
    
    def compute_effmass(self, particle, momentum = [0, 0, 0]):
        """Computes the effective mass curve for the specified particle with
        specified momentum"""
        
        if not hasattr(self, "{}_px{}_py{}_pz{}".format(particle, *momentum)):
            self.meson_correlator(particle, momentum)
        
        correlator = getattr(self, "{}_px{}_py{}_pz{}".format(particle, *momentum))
            
        return np.log(np.abs(correlator / np.roll(correlator, -1)))
    
    def __add__(self, tp):
        """Addition operator overload"""
        
        if type(tp) != type(self):
            raise TypeError("Types {} and {} do not match"
                            .format(type(self), type(tp)))
        
        for cm in self.common_members[:2]:
            if getattr(self, cm) != getattr(tp, cm):
                raise ValueError("Attribute {} differs between objects "
                                 "({} and {})".format(cm,
                                                      getattr(self, cm),
                                                      getattr(tp, cm)))
            
        new_prop1 = self.prop1 + tp.prop1
        new_prop2 = self.prop2 + tp.prop2
        
        out = TwoPoint(new_prop1, new_prop2)
        
        comp_corr1 = self.computed_correlators
        comp_corr2 = tp.computed_correlators
        
        for cc in comp_corr1:
            setattr(out, cc, getattr(self, cc))
            
        for cc in comp_corr2:
            if hasattr(out, cc):
                setattr(out, cc, getattr(out, cc) + getattr(tp, cc))
            else:
                setattr(out, cc, getattr(tp, cc))
                
        return out
    
    def __div__(self, div):
        """Division operator overloading"""
        
        if type(div) != int and type(div) != float:
            raise TypeError("Expected an int or float divisor, got {}"
                            .format(type(div)))
        
        new_prop1 = self.prop1 / div
        new_prop2 = self.prop2 / div
        
        out = TwoPoint(new_prop1, new_prop2)
        
        for cc in self.computed_correlators:
            setattr(out, cc, getattr(self, cc) / div)
        
        out.computed_correlators = self.computed_correlators
            
        return out
    
    def __neg__(self):
        """Negation operator overload"""
            
        new_prop1 = -self.prop1
        new_prop2 = -self.prop2
        
        out = TwoPoint(new_prop1, new_prop2)
        
        comp_corr1 = self.computed_correlators
        out.computed_correlators = comp_corr1
        
        for cc in comp_corr1:
            setattr(out, cc, -getattr(self, cc))
                
        return out
    
    def __sub__(self, tp):
        """Subtraction operator overload"""
        
        return self.__add__(tp.__neg__())
                        
    def __str__(self):
        
        out = \
          "Two-Point Function Object\n" \
        "-------------------------\n" \
        "Spatial extent: {}\n" \
        "Temportal extent: {}\n" \
        "Gauge action: {}\n" \
        "Inverse coupling (beta): {}\n" \
        "Mean link (u0): {}\n\n" \
        "*** Propagator 1 ***\n" \
        "Bare quark mass (m): {}\n" \
        "Inversion source site: {}\n" \
        "Number of stout field smears: {}\n" \
        "Stout smearing parameter: {}\n" \
        "Number of source Jacobi smears: {}\n" \
        "Source Jacobi smearing parameter: {}\n" \
        "Number of sink Jacobi smears: {}\n" \
        "Sink Jacobi smearing parameter: {}\n\n" \
        "*** Propagator 2 ***\n" \
        "Bare quark mass (m): {}\n" \
        "Inversion source site: {}\n" \
        "Number of stout field smears: {}\n" \
        "Stout smearing parameter: {}\n" \
        "Number of source Jacobi smears: {}\n" \
        "Source Jacobi smearing parameter: {}\n" \
        "Number of sink Jacobi smears: {}\n" \
        "Sink Jacobi smearing parameter: {}\n" \
        .format(self.prop1.L, self.prop1.T, self.prop1.action,
                self.prop1.beta, self.prop1.u0, self.prop1.mass,
                self.prop1.source_site, self.prop1.num_field_smears,
                self.prop1.field_smearing_param,
                self.prop1.num_source_smears,
                self.prop1.source_smearing_param,
                self.prop1.num_sink_smears, self.prop1.sink_smearing_param,
                self.prop2.mass, self.prop2.source_site,
                self.prop2.num_field_smears,
                self.prop2.field_smearing_param,
                self.prop2.num_source_smears,
                self.prop2.source_smearing_param,
                self.prop2.num_sink_smears, self.prop2.sink_smearing_param)
        
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
        
        Gamma1 = np.matrix(np.dot(gamma1, const.gamma5))
        Gamma2 = np.matrix(np.dot(const.gamma5, gamma2))
        
        gp1 = Gamma1 * prop1.transpose_spin().transpose_colour().conjugate()
        gp2 = Gamma2 * prop2
        
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
