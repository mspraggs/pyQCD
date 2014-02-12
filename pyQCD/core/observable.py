import numpy as np

class Observable(object):
    """Creates an observable object with the supplied data
    
    Args:
      data (optional): The data to be held by the observable object.
      
    Returns:
      Observable: The created observable object.
      
    Examples:
      Create some numpy data and encapsulate it using an observable object
      
      >>> import pyQCD
      >>> import numpy.random
      >>> data = numpy.random.random(100)
      >>> observable = pyQCD.Observable(data)
    """
    
    members = []
    
    def __init__(self, data=None):
        """Constructor for pyQCD.Observable (see help(pyQCD.Observable))"""
        
        if data == None:
            self.data = np.array([])
        else:
            self.data = data
        
    def save(self, filename):
        """Saves the observable to a numpy zip archive
        
        Args:
          filename (str): The name of the file in which to save the observable
          
        Examples:
          Create an observable object with some dummy data, then save it.
          
          >>> import pyQCD
          >>> import numpy.random
          >>> data = numpy.random(100)
          >>> observable = pyQCD.Observable(data)
          >>> observable.save("my_observable.npz")
        """
        
        items = [getattr(self, key) for key in self.members]
        
        np.savez(filename, header = dict(zip(self.members, items)),
                 data = self.data)
        
    @classmethod
    def load(cls, filename):
        """Loads and returns an observable object from a numpy zip
        archive
        
        Args:
          filename (str): The filename from which to load the observable
          
        Returns:
          Observable: The loaded observable object.
          
        Examples:
          Load an observable object from disk.
          
          >>> import pyQCD
          >>> observable = pyQCD.Observable.load("my_observable.npz")
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
        
        Args:
          filename (str): The filename to save the data as.
          
        Examples:
          Create and observable and save the raw data to disk.
          
          >>> import pyQCD
          >>> import numpy.random
          >>> data = numpy.random.random(100)
          >>> observable = pyQCD.Observable(data)
          >>> observable.save_raw("observable_data")
        """
        np.save(filename, self.data)
        
    def header(cls):
        """Retrieves the list of variables used in the header
        
        Returns:
          dict: The header variables encapsulated in a dictionary
          
        Examples:
          Create an empty observable object and retrieve the (empty) header
          
          >>> import pyQCD
          >>> observable = pyQCD.Observable()
          >>> observable.header()
          {}
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
    
    def __pow__(self, exponent):
        """Power operator overload"""
        
        if type(div) != int and type(div) != float:
            raise TypeError("Expected an int or float exponent, got {}"
                            .format(type(exponent)))
        
        new_data = self.data ** exponent
        
        return self.__class__(new_data, **self.header())
             
    def __str__(self):
        
        string_list = ["Observable Object\n",
                       "-----------------\n"]
            
        for member in self.members:
            string_list.append("{}: {}\n".format(member,
                                                 getattr(self, member)))
        
        return "".join(string_list)
