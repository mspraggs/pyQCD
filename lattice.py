import numpy as np
import numpy.random as npr
import scipy.linalg as spla
import itertools
import IPython
import copy

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
        
        self.links = np.zeros((n,n,n,n,4,3,3),dtype=complex)
        indices = itertools.product(range(n),range(n), \
                                    range(n),range(n),range(4))
        for i,j,k,l,m in indices:
            self.links[i,j,k,l,m,:,:] = np.eye(3)

        self.randSU3s = []

        for i in xrange(50):
            SU3 = self.randomSU3()
            self.randSU3s.append(SU3)
            self.randSU3s.append(SU3.H)

    def link(self,site,dir):
        """Returns given link variable as np matrix"""
        site = tuple([i%self.n for i in site])
        return np.matrix(self.links[site + (dir,)])

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

    def Si(self,link):
        """Calculates the contribution to the action by the given
        site"""
        planes = [i for i in range(4) if link[-1] != i]
        Psum = 0

        for plane in planes:
            sites = [copy.copy(list(link[0:-1])), \
                     copy.copy(list(link[0:-1]))]
            sites[1][plane] -= 1
            for s in sites:
                Psum += self.P(s,link[-1],plane)

        return -self.beta * Psum

    def randomSU3(self):
            """Generates random SU3 matrix"""
        
            A = npr.rand(3,3) * np.exp(1j * 2 * np.pi * npr.rand(3,3))
            B = np.eye(3) + 1j * self.eps * A        
            q,r = spla.qr(B)        
            return np.matrix(q) / spla.det(q)**(1./3)

    def update(self):
        """Iterate through the sites and update the link variables"""

        indices = itertools.product(range(self.n),range(self.n), \
                                    range(self.n),range(self.n), \
                                    range(4))

        for index in indices:
            Si_old = self.Si(index)
            linki_old = self.links[index]
            randSU3 = \
                self.randSU3s[npr.randint(0,high=len(self.randSU3s))]
            self.links[index] = \
                randSU3 * self.link(index[:-1],index[-1])
            dS = self.Si(index) - Si_old
            if dS > 0 and np.exp(-dS) < npr.rand():
                self.links[index] = linki_old
    
