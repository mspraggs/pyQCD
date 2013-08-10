
from core import lattice
from interfaces import inputxml
import sys
import time
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", action = "store", type = "string",
				  dest = "input_file")

(options,args) = parser.parse_args()

xmlinput = inputxml.Xml(options.input_file)
if xmlinput.check_root() == False:
	print("Error! XML input is missing pyQCD tag.")
	sys.exit()

lattice_settings = xmlinput.parse_lattice();
if lattice_settings == None:
	print("Error! There was an error parsing the lattice settings.")
	sys.exit()
	
simulation_settings = xmlinput.parse_simulation();
if simulation_settings == None:
	print("Error! There was an error parsing the simulation settings.")
	sys.exit()
	
gauge_action_settings = xmlinput.parse_gauge_action();
if gauge_action_settings == None:
	print("Error! There was an error parsing the gauge action settings.")
	sys.exit()

print("Creating the lattice...")
lattice = lattice.Lattice(lattice_settings['L'], #L
						  lattice_settings['T'],
						  gauge_action_settings['beta'], #beta
						  gauge_action_settings['u0'], #u0
						  gauge_action_settings['type'], #action
						  simulation_settings['measurement_spacing'], #Ncor
						  0.3, #rho
						  simulation_settings['update_method'], #updateMethod
						  simulation_settings['parallel_flag'], #parallelFlag
						  simulation_settings['block_size']) #chunkSize
print("Done!")

t0 = time.time()

#Thermalize the lattice
print("Thermalizing...")
sys.stdout.flush()
lattice.thermalize()
print("Done!")
sys.stdout.flush()

if simulation_settings['timing_flag']:
	t1 = time.time()
	print("Calculating run time...")
	sys.stdout.flush()
	for i in xrange(simulation_settings['num_timing_configs']):
		print("Configuration: %d" % i)
		sys.stdout.flush()
		lattice.next_config()
		
	t2 = time.time()
	print("Estimated run time: %f hours"
		% (((t2 - t1) / simulation_settings['num_timing_configs'] \
			* simulation_settings['num_configurations'] + t2 - t1) / 3600))

else:
	for i in xrange(simulation_settings['num_configurations']):
		print("Configuration: %d" % i)
		sys.stdout.flush()
		lattice.next_config()

	tf = time.time()
	hrs = int((tf - t0) / 3600)
	mins = int((tf - t0 - 3600 * hrs) / 60)
	secs = (tf - t0) - 3600 * hrs - 60 * mins
	print("Simulation completed in %d hours, %d minutes and %f seconds" \
		% (hrs,mins,secs))
