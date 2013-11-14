from observable import Observable
from propagator import Propagator
import numpy as np
import constants as const
import itertools
import scipy.optimize as spop

class TwoPoint(Observable):
    
    common_members = ['L', 'T', 'beta', 'u0', 'action']
    
    def __init__(self, prop1, prop2):
        """Create a two-point function from two propagators
        
        :param prop1: The first propagator in the two-point function
        :type prop1: :class:`Propagator`
        :param prop2: The second propagator in the two-point function
        :type prop2: :class:`Propagator`
        :raises: ValueError
        """
        
        for member in TwoPoint.common_members:
            prop1_member = getattr(prop1, member)
            prop2_member = getattr(prop2, member)
            
            if prop1_member != prop2_member:
                raise ValueError("{} members in propagators 1 ({}) and "
                                 "2 ({}) do not match."
                                 .format(member, prop1_member,
                                         prop2_member))
        
        self.prop1 = prop1
        self.prop2 = prop2
        self.L = prop1.L
        self.T = prop1.T
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
            header_values.append(getattr(self.prop1, member))
        
        for key in self.prop1.header().keys():
            if not key in TwoPoint.common_members:
                header_key = "".join([key, "_1"])
                header_keys.append(header_key)
                header_values.append(getattr(self.prop1, key))
        
        for key in self.prop2.header().keys():
            if not key in TwoPoint.common_members:
                header_key = "".join([key, "_2"])
                header_keys.append(header_key)
                header_values.append(getattr(self.prop2, key))

        header = dict(zip(header_keys, header_values))
        
        data_keys = []
        data_values = []
        data_keys.append("prop_1")
        data_values.append(self.prop1.data)
        data_keys.append("prop_2")
        data_values.append(self.prop2.data)
        
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
        :returns: :class:`Observable`
        """
        
        numpy_archive = np.load(filename)
        
        header = numpy_archive['header'].item()
        
        prop1 = Propagator(numpy_archive['prop_1'],
                           header['L'],
                           header['T'],
                           header['beta'],
                           header['u0'],
                           header['action'],
                           header['mass_1'],
                           header['source_site_1'],
                           header['num_field_smears_1'],
                           header['field_smearing_param_1'],
                           header['num_source_smears_1'],
                           header['source_smearing_param_1'],
                           header['num_sink_smears_1'],
                           header['sink_smearing_param_1'])
        
        prop2 = Propagator(numpy_archive['prop_2'],
                           header['L'],
                           header['T'],
                           header['beta'],
                           header['u0'],
                           header['action'],
                           header['mass_2'],
                           header['source_site_2'],
                           header['num_field_smears_2'],
                           header['field_smearing_param_2'],
                           header['num_source_smears_2'],
                           header['source_smearing_param_2'],
                           header['num_sink_smears_2'],
                           header['sink_smearing_param_2'])
        
        ret = TwoPoint(prop1, prop2)
        setattr(ret, 'L', header['L'])
        setattr(ret, 'T', header['T'])
        
        for correlator in numpy_archive.keys():
            if ['prop_1', 'prop_2', 'header'].count(correlator):
                setattr(ret, correlator, numpy_archive[correlator])
        
        return ret
    
    def available_mesons(self):
        """Returns a list of possible meson interpolators for use
        in the meson_correlator function
        
        :returns: :class:`list` of tuples, each describing the meson
        state and the gamma matrix combination associated with it"""
        
        return zip(const.mesons, const.combinations)
    
    def meson_correlator(self, mesons, momenta = [0, 0, 0],
                         average_momenta = True):
        """Computes and stores the specified meson correlator within the
        current TwoPoint object
        
        :param meson: The meson interpolater(s) to use in calculating the
        correlator
        :type meson: :class:`str` or :class:`str` of strings, possibilities
        given by available_mesons
        :param momenta: The lattice momentum to project the correlator onto
        :type momenta: :class:`list` of three :class:`int`s, or a compound
        list containing several momenta
        :param average_momenta: Determines whether equivalent momenta are
        averaged over
        :type average_momenta: :class:`bool`
        """
        if type(mesons) == str:
            mesons = [mesons]
        
        # First compile a list of all momenta we need to compute for
        all_momenta = []
        if type(momenta[0]) != list:
            momenta = [momenta]
            
        for p in momenta:
            if average_momenta:
                equiv_momenta = TwoPoint._get_all_momenta(p)
                equiv_momenta = [[2 * np.pi / self.L * n for n in p]
                                 for p in equiv_momenta]
                all_momenta.append(equiv_momenta)
            else:
                all_momenta.append([[2 * np.pi / self.L * n for n in p]])
                
        # We'll need this for calculating the exponentials
        sites = list(itertools.product(xrange(self.L),
                                       xrange(self.L),
                                       xrange(self.L)))
        
        # Now go through all momenta and all mesons and compute the
        # correlators
        for meson in mesons:
            Gamma = const.Gammas[meson]
            position_correlator = self._compute_correlator(Gamma)
            position_correlator = np.reshape(position_correlator,
                                             (self.T, self.L**3))
            for i, ps in enumerate(all_momenta):
                correlator_sum = np.zeros(self.T)
                
                for p in ps:
                    exponential_prefactors = np.exp(1j * np.dot(sites, p))
                    correlator_sum += np.dot(position_correlator,
                                             exponential_prefactors).real
                    
                correlator_sum /= len(ps)
                
                member_name = "{}_px{}_py{}_pz{}".format(meson, *momenta[i])
                if not member_name in self.computed_correlators:
                    self.computed_correlators.append(member_name)
                
                setattr(self, member_name, correlator_sum)
                
    def compute_energy(self, particles, fit_range, momenta = [0, 0, 0],
                       average_momenta = True):
        """Computes the energy of the specified particles at the specified
        momenta
        
        :param particles: The particle or list of particle to find the
        energy of
        :type particles: :class:`str` or :class:`list`
        :param fit_range: The two time slices specifying the range to fit
        over
        :type fit_range: :class:`list` of two :class:`int`s
        :param momenta: The momentum or a list of momenta to compute the
        energy for
        :type momenta: :class:`list`
        :param average_momenta: Determines whether equivalent momenta should
        be averaged over
        :type average_momenta: :class:`bool`
        :returns: :class:`dict` with keys specifying particles and momenta
        """
        
        # First make sure that all relevant correlators have been calculated
        if type(momenta[0]) != list:
            momenta = [momenta]
            
        if type(particles) != list:
            particles = [particles]
            
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

        for particle in particles:
            for momentum in momenta:
                attrib_name = "{}_px{}_py{}_pz{}".format(particle,
                                                         *momentum)
                current_correlator \
                  = getattr(self, attrib_name)
                x = np.arange(current_correlator.size)[fit_range[0]:
                                                       fit_range[1]]
                y = current_correlator[fit_range[0]:fit_range[1]]
                
                fit_function = lambda b, t, Ct: \
                  Ct - b[0] * (np.exp(-b[1] * (self.T - t))
                               + np.exp(-b[1] * t))
                  
                b, result = spop.leastsq(fit_function, [1.0, 1.0],
                                         args=(x, y))
                
                if [1, 2, 3, 4].count(result) < 1:
                    print("Warning: fit failed for {} with momentum {}"
                          .format(particle, momentum))
                    
                energies.append(b[1])
                keys.append(attrib_name)
                
        return dict(zip(keys, energies))
                
    def compute_square_energy(self, particles, fit_range, momenta = [0, 0, 0],
                              average_momenta = True):
        """Computes the square energy of the specified particles at the specified
        momenta
        
        :param particles: The particle or list of particle to find the
        energy of
        :type particles: :class:`str` or :class:`list`
        :param fit_range: The two time slices specifying the range to fit
        over
        :type fit_range: :class:`list` of two :class:`int`s
        :param momenta: The momentum or a list of momenta to compute the
        energy for
        :type momenta: :class:`list`
        :param average_momenta: Determines whether equivalent momenta should
        be averaged over
        :type average_momenta: :class:`bool`
        :returns: :class:`dict` with keys specifying particles and momenta
        """
        
        energies = self.compute_energy(particles, fit_range, momenta,
                                       average_momenta)
        
        return dict(zip(energies.keys(),
                        [x**2 for x in energies.values()]))
    
    def compute_c_square(self, particle, fit_range, momenta,
                         average_momentum=True, use_lattice_momenta=True):
        """Computes the square speed of light of the specified particles at the
        specified momenta
        
        :param particle: The particle to find the speed of light of
        :type particles: :class:`str`
        :param fit_range: The two time slices specifying the range to fit
        over
        :type fit_range: :class:`list` of two :class:`int`s
        :param momenta: The list of momenta to compute the speed of light for
        :type momenta: :class:`list`
        :param average_momenta: Determines whether equivalent momenta should
        be averaged over
        :type average_momenta: :class:`bool`
        :param use_lattice_momenta: Determines whether a sine function is used
        when implementing the dispersion relation
        :type average_momenta: :class: `bool`
        :returns: :class:`list` with entries corresponding to the supplied momenta
        """
        
        if type(momenta[0]) != list:
            momenta = [momenta]
        E0_square = self.compute_square_energy(particle, fit_range, [0, 0, 0])
        Es_square = self.compute_square_energy(particle, fit_range, momenta)
        
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
            
        return out
                        
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

    def _compute_correlator(self, gamma):
        """Calculates the correlator for all space-time points
        
        We're doing the following (g = Gamma, p1 = self.prop1,
        p2 = self.prop2, g5 = const.gamma5):
        
        sum_{spin,colour} g * g5 * p1 * g5 * g * p2
        
        The g5s are used to find the Hermitian conjugate of the first
        propagator
        """
        
        Gamma1 = np.matrix(np.dot(gamma, const.gamma5))
        Gamma2 = np.matrix(np.dot(const.gamma5, gamma))
        
        gp1 = Gamma1 * self.prop1.transpose_spin() \
          .transpose_colour().conjugate()
        gp2 = Gamma2 * self.prop2
        
        return np.einsum('txyzijab,txyzjiba->txyz',
                         gp1.data, gp2.data).real

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
