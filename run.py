import lib.lattice as lattice
import numpy as np
import fileio
import sys

L = lattice.Lattice()

#Thermalize the lattice

for i in xrange(5*L.Ncor):
    print("Thermalizing: %d" % i)
    sys.stdout.flush()
    L.update()

Ps = [0] * L.Ncf
    
for i in xrange(L.Ncf):
    for j in xrange(L.Ncor):
        print("Configuration: %d; Update: %d" % (i,j))
        sys.stdout.flush()
        L.update()
    Ps[i] = L.Pav()

fileio.writedata("plaquettes.txt",[Ps])
