from observable import Observable
from propagator import Propagator
import numpy as np
import constants as const

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
                                 .format(member, prop1_member, prop2_member))
        
        self.prop1 = prop1
        self.prop2 = prop2
        self.computed_correlators = {}

    
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
        
        return TwoPoint(prop1, prop2)
    
    def available_mesons(self):
        """Returns a list of possible meson interpolators for use
        in the meson_correlator function
        
        :returns: :class:`list` of tuples, each describing the meson
        state and the gamma matrix combination associated with it"""
        
        return zip(const.Gamma_mesons, const.Gamma_combinations)
    
        
    def __repr__(self):
        
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
        .format(self.prop1.L, self.prop1.T, self.prop1.action, self.prop1.beta,
                self.prop1.u0, self.prop1.mass, self.prop1.source_site,
                self.prop1.num_field_smears, self.prop1.field_smearing_param,
                self.prop1.num_source_smears, self.prop1.source_smearing_param,
                self.prop1.num_sink_smears, self.prop1.sink_smearing_param,
                self.prop2.mass, self.prop2.source_site,
                self.prop2.num_field_smears, self.prop2.field_smearing_param,
                self.prop2.num_source_smears, self.prop2.source_smearing_param,
                self.prop2.num_sink_smears, self.prop2.sink_smearing_param)
        
        return out
