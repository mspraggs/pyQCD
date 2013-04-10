import numpy as np
import numpy.random as npr
import scipy.linalg as spla
import itertools

class Lattice:

    def __init__(self,
                 n=8,
                 beta=5.5,
                 a=0.25,
                 Ncor=50,
                 Ncf=1000,
                 eps=0.24,):
        """Constructor"""
        self.beta = beta
        self.n = n
        self.Ncor = 50
        self.Ncf = Ncf
        self.eps = eps
        
        self.sites = np.zeros(n,n,n,n,3,3)
        indices = itertools.product(range(n),range(n),range(n),range(n))
        for i,j,k,l in indices:
            self.sites[i,j,k,l,:,:] = np.eye(3)
    
    def randomSU3(eps=0.24):
        """Generates random SU3 matrix"""
        
        A = npr.rand(3,3) * np.exp(1j * 2 * np.pi * npr.rand(3,3))
        B = np.eye(3) + 1j * eps * A        
        q,r = spla.qr(B)
        
        return np.matrix(q) / spla.det(q)**(1./3)
