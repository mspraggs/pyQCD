
class Config:
    
    def __init__(self):
        pass
    
    def save(self, filename):
        pass
    
    def __str__(self):
        pass
    
    def av_plaquette(self):
        pass

    @classmethod
    def load(self, filename):
        pass
        
    def save_raw(self, filename):
        np.save(filename, self.data)
    
    def __str__(self):
        pass
    
    def av_plaquette(self):
        pass

