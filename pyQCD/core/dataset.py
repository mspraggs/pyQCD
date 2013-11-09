import cPickle
import zipfile
import os

class DataSet:
    # Generic data set class. Contains header that denotes type of data.
    # Routines to perform statistical measurements on the entire dataset
    def __init__(self, datatype, filename, compress=True):
        """Create an empty data set holding data of the specified type
        
        :param datatype: The data type stored in the data set
        :type datatype: :class:`type`
        :param filename: The npz file to save the data to
        :type filename: :class:`str`
        """
        
        storage_mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        
        self.datatype = datatype
        self.num_data = 0
        self.filename = filename
        
        try:
            zfile = zipfile.ZipFile(filename, 'w', storage_mode, True)
            self.large_file = True
        except RuntimeError:
            print("Warning: > 2GB data set not supported.")
            storage_mode = zipfile.ZIP_STORED
            zfile = zipfile.ZipFile(filename, 'w', storage_mode, False)
            self.large_file = False
            
        self.storage_mode = storage_mode
        
        typefile = open("type", 'w')
        cPickle.dump(datatype, typefile)
        typefile.close()
        
        zfile.write("type")
        zfile.close()
        os.unlink("type")
    
    def add_datum(self, datum):
        """Adds a datum to the dataset
        
        :param datum: The data to add
        :type datum: The type specified in the object constructor
        :raises: TypeError
        """
        
        if type(datum) != self.datatype:
            raise TypeError("Supplied data type {} does not match the required "
                            "data type {}".format(type(datum), self.datatype))
        
        datum.save("{}{}".format(self.datatype.__name__, self.num_data))
        
        with zipfile.ZipFile(self.filename, 'a', self.storage_mode,
                             self.large_file) as zfile:
            zfile.write("{}{}.npz".format(self.datatype.__name__, self.num_data))
        
        os.unlink("{}{}.npz".format(self.datatype.__name__, self.num_data))
        self.num_data += 1
    
    def get_datum(self, index):
        """Retrieves the specified datum from the zip archive
        
        :param index: The index of the item to be retrieved
        :type index: :class:`int`
        :returns: The datum as the type specified in the constructor
        """
        
        filename = "{}{}.npz".format(self.datatype.__name__, index)
        
        with zipfile.ZipFile(self.filename, 'r', self.storage_mode,
                             self.large_file) as zfile:
            zfile.extract(filename)
            
        output = self.datatype.load(filename)
        os.unlink(filename)
        
        return output
    
    def measure(self, func, data=[], stddev=False, args=[]):
        """Performs a measurement on each item in the data set using the
        supplied function and returns the average of the measurements
        
        :param func: The measurement function, whose first argument is the
        datum object on which to perform the measurement
        :type func: :class:`function`
        :param data: The data indices to perform the measurement on
        :type data: :class:`list`
        :param stddev: If True, returns a list of two elements, with the
        second element being the standard deviation in the result
        :type stddev: :class:`bool`
        :param args: The arguments required by the supplied function
        :type args: :class:`list`
        """
        pass
    
    def bootstrap(self, func, binsize, num_bootstraps, args=[]):
        """Performs a bootstraped measurement on the dataset using the
        supplied function
        
        :param func: The measurement function
        :type func: :class:`function`
        :param binsize: The bin size used when binning
        :type binsize: :class:`int`
        :param num_bootstraps: The number of bootstraps to perform
        :type num_bootstraps: :class:`int`
        :param args: The arguments required by the supplied function
        :type args: :class:`list`
        """
        pass
    
    def jackknife(self, function, binsize, args):
        pass
        
    def save(self, filename):
        pass
    
    @classmethod
    def load(self, filename):
        pass

    def compute_observable(self, measurement, num_bootstraps, jacknife,
                           bin_size, args):
        
        func = lambda x: measurement(x, *args)
