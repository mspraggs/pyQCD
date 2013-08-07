
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
	print("a = %f" % options.a)
	print("rho = %f" % options.rho)
	print("n_smears = %d" % options.n_smears)
	print("u0 = %f" % options.u0)
	print("action = %d" % options.action)
	print("mass = %f" % options.mass)
	print("solver_method = %d" % options.solver_method)

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
				  default = 10)
parser.add_option("--Ncf", action = "store", type = "int", dest = "Ncf",
				  default = 1000)
parser.add_option("-s", "--spacing", action = "store", type = "float",
				  dest = "a", default = 0.25)
parser.add_option("--rho", action = "store", type = "float",
				  dest = "rho", default = 0.3)
parser.add_option("--update-method", action = "store", type = "int",
				  dest = "update_method", default = 0)
parser.add_option("--parallel-flag", action = "store", type = "int",
				  dest = "parallel_flag", default = 1)
parser.add_option("--test", "-t", action = "store", type = "int",
				  dest = "num_trials", default = 0)
parser.add_option("--store-plaquette", "-P", action = "store_true",
				  dest = "store_plaquette")
parser.add_option("--store-wloop", "-W", action = "store_true",
				  dest = "store_wloop")
parser.add_option("--store-configs", "-C", action = "store_true",
				  dest = "store_configs")
parser.add_option("--store-props", "-p", action = "store_true",
				  dest = "store_props")
parser.add_option("--solver-method", action = "store", type = "int",
				  dest = "solver_method", default = 0)
parser.add_option("--mass", action = "store", type = "float",
				  dest = "mass", default = 1.0)

(options,args) = parser.parse_args()

L = pyQCD.Lattice(options.n, #n
					options.beta, #beta
                    options.u0, #u0
					options.action, #action			
                    options.Ncor, #Ncor
                    options.rho, #rho
					options.update_method, #updateMethod
					options.parallel_flag) #parallelFlag

t0 = time.time()

printConfig(options)

#Thermalize the lattice
print("Thermalizing...")
sys.stdout.flush()
L.thermalize()
print("Done!")
sys.stdout.flush()

rmax = L.n_points
tmax = L.n_points

array_size = options.num_trials \
  if options.num_trials > 0 else options.Ncf

if options.store_wloop == True:
	Ws = np.zeros((array_size, rmax - 1, tmax - 1))
if options.store_plaquette == True:
	Pavs = np.zeros(array_size)

prop_shape = (array_size, L.n_points**4, 12, 12)
	
if options.store_props == True:
	props = np.zeros(prop_shape, dtype=complex)

config_shape = (array_size, L.n_points,
				L.n_points, L.n_points,
				L.n_points, 4, 3, 3)

if options.store_configs == True:
	configs = np.zeros(config_shape, dtype=complex)

if options.num_trials > 0:
	t1 = time.time()
	print("Calculating run time...")
	sys.stdout.flush()
	for i in xrange(options.num_trials):
		print("Configuration: %d" % i)
		sys.stdout.flush()
		L.next_config()
		if options.store_wloop == True:
			Ws[i] = interfaces.get_wilson_loops(L, rmax, tmax,
												n_smears = options.n_smears)

		if options.store_configs == True:
			configs[i] = np.array(interfaces.get_links(L))

		if options.store_props == True:
			props[i] = np.array(interfaces.get_propagator(L,
														  options.mass,
														  [0,0,0,0],
														  options.a,
														  options.solver_method))
			
		if options.store_plaquette == True:
			Pavs[i] = L.av_plaquette()
			print("Average plaquette: %f" % Pavs[i])
	t2 = time.time()
	if options.store_plaquette == True:
		print("Average plaquette value: %f" % Pavs[0])
	print("Estimated run time: %f hours"
		% (((t2 - t1) / options.num_trials * options.Ncf + t2 - t1) / 3600))

else:
	for i in xrange(options.Ncf):
		print("Configuration: %d" % i)
		sys.stdout.flush()
		L.next_config()
		if options.store_wloop == True:
			Ws[i] = interfaces.get_wilson_loops(L, rmax, tmax,
												n_smears = options.n_smears)

		if options.store_props == True:
			props[i] = np.array(interfaces.get_prpagator(L,
														 options.mass,
														 [0,0,0,0],
														 options.a,
														 options.solver_method))

		if options.store_configs == True:
			configs[i] = np.array(interfaces.get_links(L))

		if options.store_plaquette == True:
			Pavs[i] = L.av_plaquette()
			print("Average plaquette: %f" % Pavs[i])

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
	if options.store_wloop == True:
		Ws_filepath = join(filepath, "Ws")
		np.save(Ws_filepath,Ws)
	if options.store_plaquette == True:
		Ps_filepath = join(filepath, "Ps")
		np.save(Ps_filepath,Pavs)
	if options.store_props == True:
		props_filepath = join(filepath, "props")
		np.save(props_filepath,props)
	if options.store_configs == True:
		configs_filepath = join(filepath, "configs")
		np.save(configs_filepath, configs)

	printConfig(options)

	tf = time.time()
	hrs = int((tf - t0) / 3600)
	mins = int((tf - t0 - 3600 * hrs) / 60)
	secs = (tf - t0) - 3600 * hrs - 60 * mins
	print("Simulation completed in %d hours, %d minutes and %f seconds" \
		% (hrs,mins,secs))
