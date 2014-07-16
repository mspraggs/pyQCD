from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import sys
import xml.etree.ElementTree as ET
import struct

import numpy as np

from . import qcdutils
from ..core.constants import *

def _extract_config(config_binary):
    
    dtype, Nt, Nx, Ny, Nz = config_binary.read_header()
    
    sites = [(t, x, y, z)
             for t in xrange(Nt)
             for x in xrange(Nx)
             for y in xrange(Ny)
             for z in xrange(Nz)]
    
    config_numpy = np.zeros((Nt, Nx, Ny, Nz, 4, 3, 3),
                            dtype=np.complex)
    
    for site in sites:
        raw_site_data = config_binary.read_data(*site)
        for mu in xrange(4):

            link_matrix = np.array(raw_site_data[18 * mu:18 * (mu + 1)])
            link_matrix = link_matrix.reshape((3, 3, 2))

            complex_link_matrix \
              = link_matrix[:, :, 0] + 1j * link_matrix[:, :, 1]

            config_numpy[site + (mu,)] = complex_link_matrix

    return config_numpy

def load_ildg_config(filename):
    """Converts the specified ILDG field configuration into a numpy array

    The resulting numpy array should have shape (T, L, L, L, 4, 3, 3)

    Args:
      filename (str): The filename of the ILDG configuration

    Returns:
      numpy.ndarray: The field configuration as a numpy array.
    """
    
    config = qcdutils.GaugeILDG(filename)

    return _extract_config(config)

def load_mdp_config(filename):
    """Converts the specified MDP field configuration into a numpy array

    The resulting numpy array should have shape (T, L, L, L, 4, 3, 3)

    Args:
      filename (str): The filename of the MDP configuration

    Returns:
      numpy.ndarray: The field configuration as a numpy array.
    """

    config = qcdutils.GaugeMDP(filename)

    return _extract_config(config)

def load_scidac_config(filename):
    """Converts the specified SCIDAC field configuration into a numpy array

    The resulting numpy array should have shape (T, L, L, L, 4, 3, 3)

    Args:
      filename (str): The filename of the SCIDAC configuration

    Returns:
      numpy.ndarray: The field configuration as a numpy array.
    """

    config = qcdutils.GaugeSCIDAC(filename)

    return _extract_config(config)

def load_milc_config(filename):
    """Converts the specified MILC field configuration into a numpy array

    The resulting numpy array should have shape (T, L, L, L, 4, 3, 3)

    Args:
      filename (str): The filename of the MILC configuration

    Returns:
      numpy.ndarray: The field configuration as a numpy array.
    """

    config = qcdutils.GaugeMILC(filename)

    return _extract_config(config)

def load_chroma_mesonspec(filename, fold=False):
    """Loads the meson correlator(s) present in the supplied Chroma mesonspec
    output xml file

    Args:
      filename (str): The name of the file in which the correlators
        are contained.
      fold (bool, optional): Determines whether the correlator is folded
        about it's mid-point.

    Returns:
      dict: Correlators indexed by particle properties.

    Examples:
      Here we load some correlators produced by Chroma's mesospec routine
      and examine the keys of the resulting dict.

      >>> import pyQCD
      >>> correlators = pyQCD.load_chroma_mesonspec("96c48_pion_corr.xml")
      >>> correlators.keys()
      [('PS_PS', (0.4, 0.4), (0, 0, 0), 'point', 'point'),
       ('PS_AX', (0.4, 0.4), (0, 0, 0), 'point', 'point')]
    """

    xmlfile = ET.parse(filename)
    xmlroot = xmlfile.getroot()

    interpolators = xmlroot.findall("Wilson_hadron_measurements/elem")

    out = {}

    for interpolator in interpolators:
        source_particle_label = interpolator.find("source_particle").text
        sink_particle_label = interpolator.find("sink_particle").text

        label = "{}_{}".format(source_particle_label,
                                   sink_particle_label)
            
        mass_1 = float(interpolator.find("Mass_1").text)
        mass_2 = float(interpolator.find("Mass_2").text)
            
        raw_source_string \
          = interpolators[0] \
          .find("SourceSinkType/elem/source_type_1").text.lower()
        raw_sink_string \
          = interpolators[0] \
          .find("SourceSinkType/elem/sink_type_1").text.lower()
            
        source_sink_types = ["point", "shell", "wall"]
            
        for source_sink_type in source_sink_types:
            if raw_source_string.find(source_sink_type) > -1:
                source_type = source_sink_type
            if raw_sink_string.find(source_sink_type) > -1:
                sink_type = source_sink_type
                    
        correlator_data = interpolator.findall("Mesons/momenta/elem")
            
        for correlator_datum in correlator_data:
                
            momentum_string = correlator_datum.find("sink_mom").text
            momentum = tuple(int(x) for x in momentum_string.split())
                
            correlator_value_elems \
              = correlator_datum.findall("mesprop/elem/re")
                  
            correlator = np.array([float(x.text)
                                   for x in correlator_value_elems])

            out[(label, (mass_1, mass_2), momentum, source_type, sink_type)] \
              = fold_correlator(correlator) if fold else correlator
              
    return out

def load_chroma_hadspec(filename, fold=False):
    """Loads the correlator(s) present in the supplied Chroma hadspec output
    xml file
        
    Args:
      filename (str): The name of the file in which the correlators
        are contained.
      fold (bool, optional): Determines whether the correlator is folded
        about it's mid-point.

    Returns:
      dict: Correlators indexed by particle properties.
            
    Examples:
      Load some correlators computed by Chroma's hadspec routine.
          
      >>> import pyQCD
      >>> correlators = pyQCD.load_chroma_hadspec("96c48_hadspec_corr.xml")
    """

    output = {}

    for key, value in load_chroma_hadspec_mesons(filename, fold).items():
        output[key] = value
    for key, value in load_chroma_hadspec_baryons(filename, fold).items():
        output[key] = value
    for key, value in load_chroma_hadspec_currents(filename, fold).items():
        output[key] = value

    return output
            
def load_chroma_hadspec_mesons(filename, fold=False):
    """Loads the meson correlator(s) present in the supplied Chroma hadspec
    output xml file
        
    Args:
      filename (str): The name of the file in which the correlators
        are contained.
      fold (bool, optional): Determines whether the correlator is folded
        about it's mid-point.

    Returns:
      dict: Correlators indexed by particle properties.
            
    Examples:
      Load some correlators computed by Chroma's hadspec routine.
          
      >>> import pyQCD
      >>> correlators \
      ...   = pyQCD.load_chroma_hadspec_mesons("96c48_hadspec_corr.xml")
    """
        
    xmlfile = ET.parse(filename)
    xmlroot = xmlfile.getroot()
        
    propagator_pairs = xmlroot.findall("Wilson_hadron_measurements/elem")

    out = {}
        
    for propagator_pair in propagator_pairs:            
        mass_1 = float(propagator_pair.find("Mass_1").text)
        mass_2 = float(propagator_pair.find("Mass_2").text)
            
        raw_source_string \
          = propagator_pairs[0] \
          .find("SourceSinkType/source_type_1").text.lower()
        raw_sink_string \
          = propagator_pairs[0] \
          .find("SourceSinkType/sink_type_1").text.lower()
            
        source_sink_types = ["point", "shell", "wall"]
            
        for source_sink_type in source_sink_types:
            if raw_source_string.find(source_sink_type) > -1:
                source_type = source_sink_type
            if raw_sink_string.find(source_sink_type) > -1:
                sink_type = source_sink_type
            
        interpolator_tag_prefix \
          = "{}_{}".format(source_type.capitalize(), sink_type.capitalize())
            
        interpolators \
          = propagator_pair.findall("{}_Wilson_Mesons/elem"
                                    .format(interpolator_tag_prefix))
            
        for interpolator in interpolators:
                
            gamma_matrix \
              = int(interpolator.find("gamma_value").text)
            label = mesons[gamma_matrix]
                
            correlator_data \
              = interpolator.find("momenta")
                
            for correlator_datum in correlator_data:
                momentum_string = correlator_datum.find("sink_mom").text
                momentum = tuple(int(x) for x in momentum_string.split())
                
                correlator_values = correlator_datum.findall("mesprop/elem/re")
                  
                correlator = np.array([float(x.text)
                                       for x in correlator_values])

                out[(label, (mass_1, mass_2), momentum,
                     source_type, sink_type)] \
                  = fold_correlator(correlator) if fold else correlator

    return out
            
def load_chroma_hadspec_baryons(filename, fold=False):
    """Loads the current correlator(s) present in the supplied Chroma
    hadspec output xml file
        
    Args:
      filename (str): The name of the file in which the correlators
        are contained.
      fold (bool, optional): Determines whether the correlator is folded
        about it's mid-point.

    Returns:
      dict: Correlators indexed by particle properties.
                      
    Examples:
      Load some correlators computed by Chroma's hadspec routine.
          
      >>> import pyQCD
      >>> correlators \
      ...   = pyQCD.load_chroma_hadspec_baryons("96c48_hadspec_corr.xml")
    """
        
    xmlfile = ET.parse(filename)
    xmlroot = xmlfile.getroot()
        
    propagator_pairs = xmlroot.findall("Wilson_hadron_measurements/elem")

    out = {}
        
    for propagator_pair in propagator_pairs:            
        mass_1 = float(propagator_pair.find("Mass_1").text)
        mass_2 = float(propagator_pair.find("Mass_2").text)
            
        if mass_1 == mass_2:
            baryon_names = baryons_degenerate
        elif mass_1 < mass_2:
            baryon_names = baryons_m1m2
        else:
            baryon_names = baryons_m2m1
            
        raw_source_string \
          = propagator_pairs[0] \
          .find("SourceSinkType/source_type_1").text.lower()
        raw_sink_string \
          = propagator_pairs[0] \
          .find("SourceSinkType/sink_type_1").text.lower()
            
        source_sink_types = ["point", "shell", "wall"]
            
        for source_sink_type in source_sink_types:
            if raw_source_string.find(source_sink_type) > -1:
                source_type = source_sink_type
            if raw_sink_string.find(source_sink_type) > -1:
                sink_type = source_sink_type
            
        interpolator_tag_prefix \
          = "{}_{}".format(source_type.capitalize(), sink_type.capitalize())
            
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
                momentum = tuple(int(x) for x in momentum_string.split())
                
                correlator_values \
                  = correlator_datum.findall("barprop/elem/re")
                  
                correlator = np.array([float(x.text)
                                       for x in correlator_values])

                out[(label, (mass_1, mass_2), momentum,
                     source_type, sink_type)] \
                  = fold_correlator(correlator) if fold else correlator
                
    return out
            
def load_chroma_hadspec_currents(filename, fold=False):
    """Loads the current correlator(s) present in the supplied Chroma
    hadspec output xml file
        
    Args:
      filename (str): The name of the file in which the correlators
        are contained.
      fold (bool, optional): Determines whether the correlator is folded
        about it's mid-point.

    Returns:
      dict: Correlators indexed by particle properties.
            
    Examples:
      Load some correlators computed by Chroma's hadspec routine.
          
      >>> import pyQCD
      >>> correlators \
      ...   = pyQCD.load_chroma_hadspec_currents("96c48_hadspec_corr.xml")
    """
        
    xmlfile = ET.parse(filename)
    xmlroot = xmlfile.getroot()
        
    propagator_pairs = xmlroot.findall("Wilson_hadron_measurements/elem")

    out = {}
    
    for propagator_pair in propagator_pairs:            
        mass_1 = float(propagator_pair.find("Mass_1").text)
        mass_2 = float(propagator_pair.find("Mass_2").text)
            
        raw_source_string \
          = propagator_pairs[0] \
          .find("SourceSinkType/source_type_1").text.lower()
        raw_sink_string \
          = propagator_pairs[0] \
          .find("SourceSinkType/sink_type_1").text.lower()
            
        source_sink_types = ["point", "shell", "wall"]
            
        for source_sink_type in source_sink_types:
            if raw_source_string.find(source_sink_type) > -1:
                source_type = source_sink_type
            if raw_sink_string.find(source_sink_type) > -1:
                sink_type = source_sink_type
            
        interpolator_tag_prefix \
          = "{}_{}".format(source_type.capitalize(), sink_type.capitalize())
            
        vector_currents \
          = propagator_pair.findall("{}_Meson_Currents/Vector_currents/elem"
                                    .format(interpolator_tag_prefix))
            
        for vector_current in vector_currents:
                
            current_num \
              = int(vector_current.find("current_value").text)
            label = vector_currents[current_num]
                
            correlator_data = vector_current.find("vector_current").text
                
            correlator \
              = np.array([float(x) for x in correlator_data.split()])

            out[(label, (mass_1, mass_2), (0, 0, 0), source_type, sink_type)] \
              = fold_correlator(correlator) if fold else correlator
            
        axial_currents \
          = propagator_pair.findall("{}_Meson_Currents/Axial_currents/elem"
                                    .format(interpolator_tag_prefix))
            
        for axial_current in axial_currents:
                
            current_num \
              = int(axial_current.find("current_value").text)
            label = axial_currents[current_num]
                
            correlator_data = axial_current.find("axial_current").text
                
            correlator \
              = np.array([float(x) for x in correlator_data.split()])

            out[(label, (mass_1, mass_2), (0, 0, 0), source_type, sink_type)] \
              = fold_correlator(correlator) if fold else correlator

    return out

def load_chroma_mres(filename, fold=False):
    """Loads the domain wall mres data from the provided chroma output xml
    file.
        
    The data is imported as two correlators: the pseudoscalar correlator at
    the edges of the fifth dimension (<J5a P> )and the midpoint-pseudoscalar
    correlator in the centre of the fifth dimension (<J5qa P>). The resulting
    correlators are labelled 'J5a' and 'J5qa', respectively.
        
    Args:
      filename: (str): The name of the file from which to import the
        correlators.
      fold (bool, optional): Determines whether the correlators should
        be folded about the centre of the temporal axis after they are
        imported.

    Returns:
      dict: The two correlators used to compute the residual mass.
            
    Examples:
      Here we simply load the correlators from a the supplied xml file.
      Note that the mass of each quark is also extracted, and so can
      be used when referring to results for a specific propagator
          
      >>> import pyQCD
      >>> mres_data = pyQCD.load_chroma_mres('results.out.xml')
      >>> J5a_mq0p1 = mres_data[('J5a', (0.1, 0.1))]
      >>> J5a_mq0p3 = mres_data['J5a', (0.3, 0.3))]
    """
        
    xmltree = ET.parse(filename)
    xmlroot = xmltree.getroot()
        
    propagator_roots = xmlroot.findall("InlineObservables/elem/propagator")

    out = {}
    
    for prop_root in propagator_roots:
        mass = float(prop_root.find("Input/Param/FermionAction/Mass").text)
        pseudo_pseudo_string \
          = prop_root.find("DWF_QuarkProp4/DWF_Psuedo_Pseudo/mesprop").text
        midpoint_pseudo_string \
          = prop_root.find("DWF_QuarkProp4/DWF_MidPoint_Pseudo/mesprop").text

        pseudo_pseudo_array \
          = np.array([float(x) for x in pseudo_pseudo_string.split()])
        midpoint_pseudo_array \
          = np.array([float(x) for x in midpoint_pseudo_string.split()])
              
        out["J5a", (mass, mass)] \
          = fold_correlator(pseudo_pseudo_array) \
          if fold else pseudo_pseudo_array
              
        out["J5qa", (mass, mass)] \
          = fold_correlator(midpoint_pseudo_array) \
          if fold else midpoint_pseudo_array

    return out
         
def load_ukhadron_mesbin(filename, byteorder, fold=False):
    """Loads the correlators contained in the specified UKHADRON binary file.
    The correlators are labelled using the gamma matrix combinations used as
    interpolators when doing the meson contractions (see pyQCD.iterpolators).
    
    Args:
      filename (str): The name of the file containing the data
        byteorder (str): The endianness of the binary file. Can either be
        "little" or "big". For data created from simulations on an intel
        machine, this is likely to be "little". For data created on one
        of the IBM Bluegene machines, this is likely to be "big".
      fold (bool, optional): Determines whether the correlators should
        be folded prior to being imported.

    Returns:
      dict: Correlators indexed by particle properties
        
    Examples:
      Import the data contained in meson_m_0.45_m_0.45_Z2.280.bin
          
      >>> import pyQCD
      >>> correlators \
      ...   = pyQCD.load_ukhadron_mesbin("meson_m_0.45_m_0.45_Z2.280.bin",
      ...                                "big")
    """
        
    if sys.byteorder == byteorder:
        switch_endianness = lambda x: x
    else:
        switch_endianness = lambda x: x[::-1]
        
    binary_file = open(filename, "rb")
        
    mom_num_string = switch_endianness(binary_file.read(4))
    mom_num = struct.unpack('i', mom_num_string)[0]

    out = {}

    for i in range(mom_num):
        header_string = switch_endianness(binary_file.read(40))
        header = switch_endianness(struct.unpack("<iiiiiiiiii", header_string))
        px, py, pz, Nmu, Nnu, T = header[4:]

        correlators = np.zeros((Nmu, Nnu, T), dtype=np.complex)
    
        for t, mu, nu in [(x, y, z) for x in range(T) for y in range(Nmu)
                          for z in range(Nnu)]:
            double_string = switch_endianness(binary_file.read(8))
            correlators[mu, nu, t] = struct.unpack('d', double_string)[0]
            double_string = switch_endianness(binary_file.read(8))
            correlators[mu, nu, t] += 1j * struct.unpack('d', double_string)[0]
                        
        for mu, nu in [(x, y) for x in range(Nmu) for y in range(Nnu)]:
            out[(interpolators[mu],
                 interpolators[nu],
                 (px, py, pz))] = correlators[mu, nu]

    return out            

def load_ukhadron_mres(filename, fold=False):
    """Loads the domain wall mres data from the provided ukhadron output xml
    file.
        
    The data is imported as two correlators: the pseudoscalar correlator at the
    edges of the fifth dimension (<J5a P>) and the midpoint-pseudoscalar
    correlator in the centre of the fifth dimension (<J5qa P>). The resulting
    correlators are labelled 'J5a' and 'J5qa', respectively.
        
    Args:
      filename (str): The name of the file from which to import the
        correlators.
      fold (bool, optional): Determines whether the correlators should be folded
        about the centre of the temporal axis after they are imported.

    Returns:
      dict: Contains the two correlators used to compute mres.

    Examples:
      Here we simply load the correlators from a the supplied xml file.
    
      >>> import pyQCD
      >>> correlators = pyQCD.load_ukhadron_mres('prop1.xml')
      >>> J5a_mq0p1 = correlators['J5a']
    """
    
    xmltree = ET.parse(filename)
    xmlroot = xmltree.getroot()
    
    dwf_observables = xmlroot.find("DWF_observables")

    out = {}
    
    pseudo_pseudo_string = dwf_observables.find("PP").text
    midpoint_pseudo_string = dwf_observables.find("PJ5q").text

    pseudo_pseudo_array \
      = np.array([float(x) for x in pseudo_pseudo_string.split()])
    midpoint_pseudo_array \
      = np.array([float(x) for x in midpoint_pseudo_string.split()])

    out["J5a"] = pseudo_pseudo_array

    out["J5qa"] = midpoint_pseudo_array

    return out
