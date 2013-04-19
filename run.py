import lib.lattice as lattice
import numpy as np
import fileio
import sys

L = lattice.Lattice()

#Thermalize the lattice

print("Thermalizing...")
sys.stdout.flush()
L.thermalize()
print("Done!")
sys.stdout.flush()

Ps = [0] * L.Ncf
    
for i in xrange(L.Ncf):
    L.nextConfig()
    print("Configuration: %d" % i)
    Ps[i] = L.Pav()

fileio.writedata("plaquettes.txt",[Ps])
