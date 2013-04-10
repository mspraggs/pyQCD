import numpy as np
import numpy.random as npr
import scipy.linalg as spla
import itertools
import IPython

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
        
        self.sites = np.zeros((n,n,n,n,4,3,3))
        indices = itertools.product(range(n),range(n), \
                                    range(n),range(n),range(4))
        for i,j,k,l,m in indices:
            self.sites[i,j,k,l,m,:,:] = np.eye(3)

    def link(self,site,dir):
        """Returns given link variable as np matrix"""
        site = tuple([i%self.n for i in site])
        return np.matrix(self.sites[site + (dir,)])

    def P(self,site,mu,nu):
        """Calculates a single plaquette"""
        #Create some arrays for the directions we've been given
        muv = nuv = np.zeros(4,dtype=int)
        muv[mu] = nuv[nu] = 1
        site = np.array(site)
        
        product = self.link(tuple(site),mu)
        product *= self.link(tuple(site + muv),nu)
        product *= self.link(tuple(site + nuv),mu).H
        product *= self.link(tuple(site), nu).H
        return 1./3 * np.real(np.trace(product))

    def Si(site):
        """Calculates the contribution to the action by the given
        site"""
        
    
    def randomSU3(self):
        """Generates random SU3 matrix"""
        
        A = npr.rand(3,3) * np.exp(1j * 2 * np.pi * npr.rand(3,3))
        B = np.eye(3) + 1j * self.eps * A        
        q,r = spla.qr(B)
        
        return np.matrix(q) / spla.det(q)**(1./3)
