import core.lattice
import interfaces.xmlinput
import interfaces.lattice
import interfaces.measurements
import sys
import time
from optparse import OptionParser
import IPython

def main():
	"""Main execution function"""
	parser = OptionParser()
	parser.add_option("-i", "--input", action = "store", type = "string",
					  dest = "input_file")
	
	(options,args) = parser.parse_args()

	xmlinput = interfaces.xmlinput.Xml(options.input_file)
	if xmlinput.check_root() == False:
		print("Error! XML input is missing pyQCD tag.")
		sys.exit()

	lattice_settings = xmlinput.parse_lattice()
	if lattice_settings == None:
		print("Error! There was an error parsing the lattice settings.")
		sys.exit()
	
	simulation_settings = xmlinput.parse_simulation()
	if simulation_settings == None:
		print("Error! There was an error parsing the simulation settings.")
		sys.exit()
	
	gauge_action_settings = xmlinput.parse_gauge_action()
	if gauge_action_settings == None:
		print("Error! There was an error parsing the gauge action settings.")
		sys.exit()

	measurement_settings = xmlinput.parse_measurements()

	print("Creating the lattice...")
	lattice \
	  = core.lattice.Lattice(lattice_settings['L'],
							 lattice_settings['T'],
							 gauge_action_settings['beta'],
							 gauge_action_settings['u0'],
							 gauge_action_settings['type'],
							 simulation_settings['measurement_spacing'],
							 simulation_settings['update_method'],
							 simulation_settings['parallel_update']['enabled'],
							 simulation_settings['parallel_update'] \
							 ['block_size'])
	print("Done!")

	t0 = time.time()

	# Create somewhere to put the measurements
	measurements = interfaces.measurements.create(measurement_settings,
												  lattice_settings,
												  simulation_settings)
	# Interface object to handle numpy types etc
	lattice_interface = interfaces.lattice.LatticeInterface(lattice)
	
	# Thermalize the lattice
	print("Thermalizing...")
	sys.stdout.flush()
	lattice.thermalize()
	print("Done!")
	sys.stdout.flush()

	num_configs = simulation_settings['timing_run']['num_configurations'] \
	  if simulation_settings['timing_run']['enabled'] \
	  else simulation_settings['num_configurations']
  
	t1 = time.time()
	print("Calculating run time...")
	sys.stdout.flush()
	for i in xrange(num_configs):
		print("Configuration: %d" % i)
		sys.stdout.flush()
		lattice.next_config()
		interfaces.measurements.do(measurement_settings,
								   lattice_interface,
								   measurements, i)

	# Store the measurments
	interfaces.measurements.save(measurement_settings, measurements)
		
	t2 = time.time()
	if simulation_settings['timing_run']['enabled']:
		total_time \
		  = (t2 - t1) / simulation_settings['timing_run']['num_configurations'] \
		  * simulation_settings['num_configurations'] + t2 - t1
	else:
		total_time = t2 - t0
		
	hrs = int((total_time) / 3600)
	mins = int((total_time - 3600 * hrs) / 60)
	secs = total_time - 3600 * hrs - 60 * mins
	
	if simulation_settings['timing_run']['enabled']:
		print("Estimated run time: %d hours, %d minutes and %f seconds" \
			  % (hrs,mins,secs))
	else:
		print("Simulation completed in %d hours, %d minutes and %f seconds" \
			  % (hrs,mins,secs))

if __name__ == "__main__":
	main()
