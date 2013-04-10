import lattice
import numpy as np

L = lattice.Lattice()

#Thermalize the lattice

for i in xrange(5*L.Ncor):
    print("Thermalizing: %d" % i)
    L.update()

Ps = np.zeros(L.Ncf)
    
for i in xrange(L.Ncf):
    for j in xrange(L.Ncor):
        print("Configuration: %d; Update: %d" % (i,j))
        L.update()
    Ps[i] = L.Pav()
