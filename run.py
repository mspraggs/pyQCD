import lib.lattice as lattice
import interfaces
import numpy as np
import fileio
import sys
import time
import datetime
from os.path import join

L = lattice.Lattice()

#Thermalize the lattice
print("Thermalizing...")
sys.stdout.flush()
L.thermalize()
print("Done!")
sys.stdout.flush()

rmax = 7
tmax = 7
Ws = np.zeros((L.Ncf,rmax-1,tmax-1))
    
for i in xrange(L.Ncf):
    print("Configuration: %d" % i)
    L.nextConfig()
    Ws[i] = interfaces.calcWs(L,rmax,tmax)

time = datetime.datetime.now()
filename = join("results","results_%s.npy" % time.strftime("%H:%M:%S_%d-%m-%Y"))
np.save(filename,Ws)
