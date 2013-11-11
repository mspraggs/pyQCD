import cPickle
import zipfile
import os
import numpy as np
import numpy.random as npr

class DataSet:
    # Generic data set class. Contains header that denotes type of data.
    # Routines to perform statistical measurements on the entire dataset
    def __init__(self, datatype, filename, compress=True):
        """Create an empty data set holding data of the specified type
        
        :param datatype: The data type stored in the data set
        :type datatype: :class:`type`
        :param filename: The zip file to save the data to
        :type filename: :class:`str`
        :param compress: Determines whether compression is used
        :type compress: :class:`bool`
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

        if data == []:
            data = range(self.num_data)
            
        measurements = []
        for i in data:
            datum = self.get_datum(i)
            measurement = func(datum, *args)
            measurements.append(measurement)
            
        if stddev:
            return DataSet._mean(measurements), DataSet._std(measurements)
        else:
            return DataSet._mean(measurements)
    
    def bootstrap(self, func, num_bootstraps, binsize=1, args=[]):
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
        
        if binsize < 1:
            raise ValueError("Supplied bin size {} is less than 1"
                             .format(binsize))
        
        num_bins = self.num_data / binsize
        if self.num_data % binsize > 0:
            num_bins += 1
            
        out = []
        
        for i in xrange(num_bootstraps):
            
            bins = npr.randint(num_bins, size = num_bins).tolist()
            
            bin_out = []            
            for b in bins:
                datum = self._get_bin(binsize, b)
                
                measurement = func(datum, *args)
                bin_out.append(measurement)
                
            out.append(DataSet._mean(bin_out))
            
            
        return DataSet._mean(out), DataSet._std(out)
    
    def jackknife(self, func, binsize, args):
        pass
        
    def save(self, filename):
        pass
    
    @classmethod
    def load(self, filename):
        pass

        
        func = lambda x: measurement(x, *args)
