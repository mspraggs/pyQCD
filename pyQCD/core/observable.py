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
        
    def __repr__(self):
        
        string_list = ["Observable Object\n",
                       "-----------------\n"]
            
        for member in self.members:
            string_list.append("{}: {}\n".format(member,
                                                 getattr(self, member)))
        
        return "".join(string_list)
