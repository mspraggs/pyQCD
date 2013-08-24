from pyQCD.core.lattice import Lattice
from pyQCD.interfaces.input import XmlInterface
from pyQCD.interfaces.lattice import LatticeInterface
import pyQCD.interfaces.measurements as measure
import sys
import time
from optparse import OptionParser

def main(input_file):
	"""Runs a simulation as specified in the supplied xml input file."""
	# Try to parse the supplied xml input file, exit if it fails
	try:
		xml = XmlInterface(input_file)
	except:
		print("Error parsing XML file.")
		sys.exit()
	# Copy the various settings to a set of convenient variables
	lattice_settings = xml.lattice()
	simulation_settings = xml.simulation()	
	gauge_action_settings = xml.gauge_action()

	# Print out the input xml
	print("Input XML:")
	print(xml)
	print("")
	sys.stdout.flush()

	# Get the measurement settings
	measurement_settings = xml.measurements()

	# Declare and initialize the lattice
	print("Creating the lattice...")
	lattice = Lattice(lattice_settings['L'],
					  lattice_settings['T'],
					  gauge_action_settings['beta'],
					  gauge_action_settings['u0'],
					  gauge_action_settings['type'],
					  simulation_settings['measurement_spacing'],
					  simulation_settings['update_method'],
					  simulation_settings['parallel_update']['enabled'],
					  simulation_settings['parallel_update']['block_size'])
	print("Done!")

	# We're going to time the run, so get the initial time
	t0 = time.time()

	# Create somewhere to put the measurements
	measurements = measure.create(measurement_settings,
								  lattice_settings,
								  simulation_settings)
	# Interface object to handle numpy types etc
	lattice_interface = LatticeInterface(lattice)
	
	# Thermalize the lattice
	print("Thermalizing...")
	sys.stdout.flush()
	lattice.thermalize()
	print("Done!")
	sys.stdout.flush()

	# Get the actual number of configs we'll be generating, for timing
	# purposes
	num_configs = simulation_settings['timing_run']['num_configurations'] \
	  if simulation_settings['timing_run']['enabled'] \
	  else simulation_settings['num_configurations']

	# For estimating the run time, split the thermalization and config
	# generation processes
	t1 = time.time()
	sys.stdout.flush()
	# Run through and do the updates
	for i in xrange(num_configs):
		print("Configuration: %d" % i)
		sys.stdout.flush()
		print("Updating gauge field...")
		sys.stdout.flush()
		lattice.next_config()
		print("Done!")
		print("Doing measurements:")
		sys.stdout.flush()
		measure.do(measurement_settings,
				   lattice_interface,
				   measurements, i)
		print("Done!")

	# Store the measurments
	if not simulation_settings['timing_run']['enabled']:
		measure.save(measurement_settings, measurements)
	# Get the final time, then calculate the total time, either
	# estimated or otherwise.
	t2 = time.time()
	if simulation_settings['timing_run']['enabled']:
		total_time \
		  = (t2 - t1) / simulation_settings['timing_run']['num_configurations'] \
		  * simulation_settings['num_configurations'] + t2 - t1
	else:
		total_time = t2 - t0
	# Convert the answer to hrs, mins, secs
	hrs = int((total_time) / 3600)
	mins = int((total_time - 3600 * hrs) / 60)
	secs = total_time - 3600 * hrs - 60 * mins

	# Print the output
	if simulation_settings['timing_run']['enabled']:
		print("Estimated run time: %d hours, %d minutes and %f seconds" \
			  % (hrs,mins,secs))
	else:
		print("Simulation completed in %d hours, %d minutes and %f seconds" \
			  % (hrs,mins,secs))

if __name__ == "__main__":
	# Parse the command line arguments
	parser = OptionParser()
	parser.add_option("-i", "--input", action = "store", type = "string",
					  dest = "input_file")
	(options,args) = parser.parse_args()
	main(options.input_file)
