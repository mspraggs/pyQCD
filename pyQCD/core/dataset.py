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
        """Load an existing data set from the supplied zip archive
        
        :param filename: The file containing the dataset
        :type filename: :class:`str`
        :returns: :class:`DataSet`
        """
        
        storage_mode = zipfile.ZIP_DEFLATED
        compress = True
        
        try:
            zfile = zipfile.ZipFile(filename, 'r', storage_mode, True)
        except RuntimeError:
            storage_mode = zipfile.ZIP_STORED
            zfile = zipfile.ZipFile(filename, 'r', storage_mode, False)
            compress = False
        
        zfile.extract("type")
        
        typefile = open("type", 'r')
        datatype = cPickle.load(typefile)
        typefile.close()
        os.unlink("type")
        
        out = DataSet(datatype, "000.zip", compress)
        out.filename = filename
        os.unlink("000.zip")
        
        data = [int(fname[len(datatype.__name__):-4])
                for fname in zfile.namelist()
                if fname.startswith(datatype.__name__)]
        
        if len(data) > 0:
            out.num_data = max(data) + 1
        
        return out
    
    def _get_bin(self, binsize, binnum):
        """Average the binsize data in binnum"""
        
        out = self.get_datum(binsize * binnum)
        
        first_datum = binsize * binnum + 1
        last_datum = binsize * (binnum + 1)
        
        if last_datum > self.num_data:
            last_datum = self.num_data
        
        for i in xrange(first_datum, last_datum):
            out += self.get_datum(i)
            
        return out / binsize

    @staticmethod
    def _add_measurements(a, b):
        """Adds two measurements (used for dictionaries etc)"""
        
        if type(a) == tuple:
            a = list(a)
        if type(b) == tuple:
            b = list(b)
        
        if type(a) == list and type(b) == list:
            return [DataSet._add_measurements(x, y) for x, y in zip(a, b)]
        elif type(a) == dict and type(b) == dict:
            return DataSet._add_measurements(a, b.values())
        elif type(a) == dict and type(b) == list:
            return dict(zip(a.keys(), DataSet._add_measurements(a.values(), b)))
        elif type(a) == list and type(b) == dict:
            return DataSet._add_measurements(b, a)
        elif (type(a) == int or type(a) == float or type(a) == np.float64) \
          and (type(b) == int or type(b) == float or type(b) == np.float64):
            return a + b
        else:
            raise TypeError("Supplied types {} and {} cannot be summed"
                            .format(type(a), type(b)))

    @staticmethod
    def _sub_measurements(a, b):
        """Adds two measurements (used for dictionaries etc)"""
        
        if type(a) == tuple:
            a = list(a)
        if type(b) == tuple:
            b = list(b)
        
        if type(a) == list and type(b) == list:
            return [DataSet._sub_measurements(x, y) for x, y in zip(a, b)]
        elif type(a) == dict and type(b) == dict:
            return DataSet._sub_measurements(a, b.values())
        elif type(a) == dict and type(b) == list:
            return dict(zip(a.keys(), DataSet._sub_measurements(a.values(), b)))
        elif type(a) == list and type(b) == dict:
            return DataSet._sub_measurements(b, a)
        elif (type(a) == int or type(a) == float or type(a) == np.float64) \
          and (type(b) == int or type(b) == float or type(b) == np.float64):
            return a - b
        else:
            raise TypeError("Supplied types {} and {} cannot be summed"
                            .format(type(a), type(b)))

    @staticmethod
    def _mul_measurements(a, b):
        """Adds two measurements (used for dictionaries etc)"""
        
        if type(a) == tuple:
            a = list(a)
        if type(b) == tuple:
            b = list(b)
        
        if type(a) == list and type(b) == list:
            return [DataSet._mul_measurements(x, y) for x, y in zip(a, b)]
        elif type(a) == dict and type(b) == dict:
            return DataSet._mul_measurements(a, b.values())
        elif type(a) == dict and type(b) == list:
            return dict(zip(a.keys(), DataSet._mul_measurements(a.values(), b)))
        elif type(a) == list and type(b) == dict:
            return DataSet._mul_measurements(b, a)
        elif (type(a) == int or type(a) == float or type(a) == np.float64) \
          and (type(b) == int or type(b) == float or type(b) == np.float64):
            return a * b
        else:
            raise TypeError("Supplied types {} and {} cannot be summed"
                            .format(type(a), type(b)))
            
    @staticmethod
    def _div_measurements(a, div):
        """Divides a measurement by a scalar value"""
        
        if type(div) != float and type(div) != int:
            raise TypeError("Unsupported divisor of type {}".format(type(div)))
        
        if type(a) == list or type(a) == tuple:
            return [DataSet._div_measurements(x, div) for x in a]
        
        if type(a) == dict:
            return dict(zip(a.keys(), DataSet._div_measurements(a.values(),
                                                                div)))
        
        if type(a) == float or type(a) == int or type(a) == np.float64:
            return a / div
            
    @staticmethod
    def _sqrt_measurements(a):
        """Divides a measurement by a scalar value"""
        
        if type(a) == list or type(a) == tuple:
            return [DataSet._sqrt_measurements(x) for x in a]
        
        if type(a) == dict:
            return dict(zip(a.keys(), DataSet._sqrt_measurements(a.values())))
        
        if type(a) == float or type(a) == int or type(a) == np.float64:
            return np.sqrt(a)
