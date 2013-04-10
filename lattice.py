import numpy as np
import numpy.random as npr
import scipy.linalg as spla

class Lattice:

    def __init__(n=8,
                 beta=5.5,
                 a=0.25,
                 Ncor=50,
                 N=20,
                 Ncf=1000,
                 eps=0.24,):
        """Constructor"""
        self.sites = np.zeros(n,n,n,n,3,3)
        
    
    def randomSU3(eps=0.24):
        """Generates random SU3 matrix"""
        
        A = -1 + 2 * npr.rand(3,3) + 1j * (-1 + 2 * npr.rand(3,3))
        B = np.eye(3) + 1j * eps * A        
        q,r = spla.qr(B)
        
        return np.matrix(q) / spla.det(q)**(1./3)
