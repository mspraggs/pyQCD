
class DataSet:
    # Generic data set class. Contains header that denotes type of data.
    # Routines to perform statistical measurements on the entire dataset
    def __init__(self):
        pass
    
    def add_datum(self, datum):
        pass
    
    def get_datum(self, index):
        pass
        
    def save(self, filename):
        pass
    
    @classmethod
    def load(self, filename):

    def compute_observable(self, measurement, num_bootstraps, jacknife,
                           bin_size, args):
        
        func = lambda x: measurement(x, *args)
