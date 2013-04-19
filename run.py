import lib.lattice as lattice
import interfaces
import numpy as np
import fileio
import sys
import time
import datetime
from os.path import join

L = lattice.Lattice(8, #n
                    5.5, #beta
                    50, #Ncor
                    10000, #Ncf
                    0.24, #eps
                    0.25, #a
                    1./12) #smear_eps

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
    sys.stdout.flush()
    L.nextConfig()
    Ws[i] = interfaces.calcWs(L,rmax,tmax)

time = datetime.datetime.now()
filename = join("results","results_%s.npy" % time.strftime("%H:%M:%S_%d-%m-%Y"))
np.save(filename,Ws)
