import dataset
import os

class Ensemble(dataset.DataSet):
    
    def add_config(self, configuration):
        self.add_datum(configuration)
        
