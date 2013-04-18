import lib.lattice as lattice
import numpy as np
import fileio
import sys

L = lattice.Lattice()

#Thermalize the lattice

print("Thermalizing...")
L.thermalize()
print("Done!")

Ps = [0] * L.Ncf
    
for i in xrange(L.Ncf):
    for j in xrange(L.Ncor):
        print("Configuration: %d; Update: %d" % (i,j))
        sys.stdout.flush()
        L.update()
    Ps[i] = L.Pav()

fileio.writedata("plaquettes.txt",[Ps])
