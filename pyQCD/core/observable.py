import numpy as np

class Observable(object):
    
    members = []
    
    def __init__(self):
        """Constructs an observable object"""
        
        self.data = np.array([])
        
    def save(self, filename):
        """Saves the observable to a numpy zip archive
        
        :param filename: The file to save to
        :type filename: :class:`str`
        """
        
        items = [getattr(self, key) for key in self.members]
        
        np.savez(filename, header = dict(zip(self.members, items)),
                 data = self.data)
        
    @classmethod
    def load(cls, filename):
        """Loads and returns an observable object from a numpy zip
        archive
        
        :param filename: The file to load from
        :type filename: :class:`str`
        :returns: :class:`Observable`
        """
        
        numpy_archive = np.load(filename)
        
        header = numpy_archive['header'].item()
        
        for key in cls.members:
            if not key in header.keys():
                raise ValueError("Missing value in archive header: {}"
                                 .format(key))
        
        data = numpy_archive['data']
            
        return cls(data, **header)
        
    def save_raw(self, filename):
        """Saves the data array as a numpy binary
        
        :param filename: The file to save to
        :type filename: :class:`str`
        """
        np.save(filename, self.data)
        
    def header(cls):
        """Retrieves the list of variables used in the header
        
        :returns: :class:`dict`
        """
        
        items = [getattr(cls, member) for member in cls.members]
        
        return dict(zip(cls.members, items))

    def __add__(self, ob):
        """Addition operator overload"""
        
        if type(ob) != type(self):
            raise TypeError("Types {} and {} do not match"
                            .format(type(self), type(ob)))
        
        for member in self.members:
            if getattr(self, member) != getattr(ob, member):
                raise ValueError("Attribute {} differs between objects "
                                 "({} and {})".format(member,
                                                      getattr(self, member),
                                                      getattr(ob, member)))
            
        new_data = self.data + ob.data
        
        return self.__class__(new_data, **self.header())
    
    def __div__(self, div):
        """Division operator overload"""
        
        if type(div) != int and type(div) != float:
            raise TypeError("Expected an int or float divisor, got {}"
                            .format(type(div)))
        
        new_data = self.data / div
        
        return self.__class__(new_data, **self.header())

    def __neg__(self):
        """Negation operator overload"""
            
        new_data = -self.data
        
        return self.__class__(new_data, **self.header())
    
    def __sub__(self, ob):
        """Subtraction operator overload"""
        
        return self.__add__(ob.__neg__())
             
    def __str__(self):
        
        string_list = ["Observable Object\n",
                       "-----------------\n"]
            
        for member in self.members:
            string_list.append("{}: {}\n".format(member,
                                                 getattr(self, member)))
        
        return "".join(string_list)
