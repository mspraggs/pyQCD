import lattice
import numpy as np

L = lattice.Lattice()

#Thermalize the lattice

for i in xrange(5*L.Ncor):
    L.update()

Ps = np.zeros(L.Ncf)
    
for i in xrange(L.Ncf):
    for j in xrange(L.Ncor):
        L.update()
    Ps[i] = L.Pav()
