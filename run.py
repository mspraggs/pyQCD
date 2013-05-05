import lib.pyQCD as pyQCD
import interfaces
import numpy as np
import fileio
import sys
import time
import datetime
from os.path import join
import os
from optparse import OptionParser

def printConfig(options):
    """Outputs the simulation configuration to the screen"""
    print("Configuration:")
    print("n = %d" % options.n)
    print("beta = %f" % options.beta)
    print("Ncor = %d" % options.Ncor)
    print("Ncf = %d" % options.Ncf)
    print("eps = %f" % options.eps)
    print("a = %f" % options.a)
    print("rho = %f" % options.rho)
    print("n_smears = %d" % options.n_smears)
    print("u0 = %f" % options.u0)
    print("action = %d" % options.action)

parser = OptionParser()
parser.add_option("-b", "--beta", action = "store", type = "float",
				  dest = "beta", default = 5.5)
parser.add_option("-u", "--u0", action = "store", type = "float",
				  dest = "u0", default = 1)
parser.add_option("-a", "--action", action = "store", type = "int",
				  dest = "action", default = 0)
parser.add_option("--nsmears", action = "store", type = "int",
				  dest = "n_smears", default = 0)
parser.add_option("-n", "--n", action = "store", type = "int", dest = "n",
				  default = 8)
parser.add_option("--Ncor", action = "store", type = "int", dest = "Ncor",
				  default = 50)
parser.add_option("--Ncf", action = "store", type = "int", dest = "Ncf",
				  default = 1000)
parser.add_option("-e", "--eps", action = "store", type = "float",
				  dest = "eps", default = 0.24)
parser.add_option("-s", "--spacing", action = "store", type = "float",
				  dest = "a", default = 0.25)
parser.add_option("--rho", action = "store", type = "float",
				  dest = "rho", default = 0.3)
parser.add_option("--test", "-t", action = "store_true", dest = "test")

(options,args) = parser.parse_args()

L = pyQCD.Lattice(options.n, #n
                    options.beta, #beta
                    options.Ncor, #Ncor
                    options.Ncf, #Ncf
                    options.eps, #epsilson
                    options.a, #a
                    options.rho, #rho
                    options.u0, #u0
                    options.action) #action

t0 = time.time()

printConfig(options)

#Thermalize the lattice
print("Thermalizing...")
sys.stdout.flush()
L.thermalize()
print("Done!")
sys.stdout.flush()

rmax = L.n_points - 1
tmax = L.n_points - 1
Ws = np.zeros((L.n_conf, rmax - 1, tmax - 1))
Pavs = np.zeros(L.n_conf)

if options.test:
    t1 = time.time()
    print("Calculating run time...")
    sys.stdout.flush()
    L.next_config()
    L.av_plaquette()
    interfaces.calcWs(L, rmax, tmax, n_smears = options.n_smears)
    t2 = time.time()
    print("Estimated run time: %f hours" \
		  % (((t2 - t1) * L.n_conf + t2 - t1) / 3600))

else:    
    for i in xrange(L.n_conf):
        print("Configuration: %d" % i)
        sys.stdout.flush()
        L.next_config()
        Ws[i] = interfaces.calcWs(L, rmax, tmax, n_smears = options.n_smears)
        Pavs[i] = L.av_plaquette()

    time_now = datetime.datetime.now()
    folder = \
	  "results_n=%d,beta=%f,Ncor=%d,Ncf=%d,u0=%f,action=%d,n_smears=%d_%s" \
	% (options.n,
	   options.beta,
	   options.Ncor,
	   options.Ncf,
	   options.u0,
	   options.action,
	   options.n_smears,
	   time_now.strftime("%H:%M:%S_%d-%m-%Y"))
	
    filepath = join("results", folder)
    os.makedirs(filepath)
    Ws_filepath = join(filepath, "Ws")
    Ps_filepath = join(filepath, "Ps")
    np.save(Ws_filepath,Ws)
    np.save(Ps_filepath,Pavs)

    printConfig(options)

    tf = time.time()
    hrs = int((tf - t0) / 3600)
    mins = int((tf - t0 - 3600 * hours) / 60)
    secs = (tf - t0) - 3600 * hours - 60 * minutes
    print("Simulation completed in %d hours, %d minutes and %f seconds" \
		  % (hrs,mins,secs))

    
    
