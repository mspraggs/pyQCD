from twopoint import TwoPoint
import numpy as np

class BareTwoPoint(TwoPoint):
    
    common_members = ['L', 'T']
    
    def __init__(self, T, L):
        """Create an empty two-point function to hold correlators
        for fitting
        
        :param T: The temporal extent of the correlator
        :type T: :class:`int`
        :param L: The spatial extent of the lattice
        :type T: :class:`int`
        """
        
        self.L = L
        self.T = T
        self.computed_correlators = []
        
    def save(self, filename):
        """Saves the two-point function to a numpy zip archive
        
        :param filename: The file to save to
        :type filename: :class:`str`
        """
        
        header_keys = []
        header_values = []
        
        for member in BareTwoPoint.common_members:
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
        """Loads and returns the two-point function stored in the
        supplied numpy zip archive
        
        :param filename: The file to load from
        :type filename: :class:`str`
        :returns: :class:`TwoPoint`
        """
        
        numpy_archive = np.load(filename)
        
        header = numpy_archive['header'].item()
        
        ret = BareTwoPoint(header['T'], header['L'])
        setattr(ret, "computed_correlators",
                header["computed_correlators"])
        
        for correlator in numpy_archive.keys():
            if correlator != 'header':
                setattr(ret, correlator, numpy_archive[correlator])
                
        return ret
        
    def meson_correlator(self, mesons, momenta = [0, 0, 0],
                         average_momenta = True):
        """Override the meson_correlator function from TwoPoint, as it
        won't work here without two propagators
        
        :raises: NotImplementedError
        """
        
        raise NotImplementedError("Propagators do not exist with which "
                                  "to compute correlator(s)")
    
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
        
        out = TwoPoint(self.T, self.L)
        
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
        
        out = BareTwoPoint(self.T, self.L)
        
        for cc in self.computed_correlators:
            setattr(out, cc, getattr(self, cc) / div)
            out.computed_correlators.append(cc)
            
        return out
                        
    def __str__(self):
        
        out = \
          "Bare Two-Point Function Object\n" \
        "------------------------------\n" \
        "Spatial extent: {}\n" \
        "Temportal extent: {}\n" \
        .format(self.L, self.T)
        
        return out
