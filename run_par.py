import lib.lattice as lattice
import interfaces
import numpy as np
import fileio
import sys
import time
import datetime
import os
from os.path import join
from optparse import OptionParser
import copy

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
parser = OptionParser()
parser.add_option("-b","--beta",action="store", type="float", dest="beta",default=5.5)
parser.add_option("-u","--u0",action="store", type="float", dest="u0",default=1)
parser.add_option("-a","--action",action="store", type="int", dest="action",default=0)
parser.add_option("--nsmears",action="store", type="int", dest="n_smears",default=0)
parser.add_option("-n","--n",action="store", type="int", dest="n",default=8)
parser.add_option("--Ncor",action="store", type="int", dest="Ncor",default=50)
parser.add_option("--Ncf",action="store", type="int", dest="Ncf",default=1000)
parser.add_option("-e","--eps",action="store", type="float", dest="eps",default=0.24)
parser.add_option("-s","--spacing",action="store", type="float", dest="a",default=0.25)
parser.add_option("--smeareps",action="store", type="float", dest="smear_eps",default=1./12)
parser.add_option("--test","-t",action="store_true",dest="test")

options, args = parser.parse_args()

if rank == 0:    
    L = lattice.Lattice(options.n, #n
                        options.beta, #beta
                        options.Ncor, #Ncor
                        options.Ncf, #Ncf
                        options.eps, #eps
                        options.a, #a
                        options.smear_eps, #smear_eps
                        options.u0, #u0
                        options.action) #action

    Ls = []
    for i in xrange(size):
        Ls.append(copy.deepcopy(L))

else:
    Ls = None

Ls = comm.scatter(Ls,root=0)

#Thermalize the lattice
print("Thermalizing...")
t0 = time.time()
sys.stdout.flush()
Ls.thermalize()
print("Done!")
sys.stdout.flush()

rmax = Ls.n-1
tmax = Ls.n-1
Ws = np.zeros((Ls.Ncf,rmax-1,tmax-1))
chunk = (Ls.Ncf)/size
my_start = rank * chunk
my_end = my_start + chunk

for i in xrange(my_start,my_end):
    print("Configuration: %d" % i)
    sys.stdout.flush()
    Ls.nextConfig()
    Ws[i] = interfaces.calcWs(Ls,rmax,tmax,n_smears=options.n_smears)

Ws = comm.gather(Ws,root=0)

if rank == 0:
    time = datetime.datetime.now()
    filename = "results_n=%d,beta=%f,Ncor=%d,Ncf=%d,u0=%d,action=%d,n_smears=%d_%s" % (options.n,options.beta,options.Ncor,options.Ncf,options.u0,options.action,options.n_smears,time.strftime("%H:%M:%S_%d-%m-%Y"))
    filepath = join("results",filename)
    np.save(filepath,Ws)
    os.system("git add %s" % filepath)
    os.system("git commit %s -m 'Adding results'" % filepath)


