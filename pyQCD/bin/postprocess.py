from pyQCD.interfaces.input import XmlInterface
import pyQCD.postprocessing.main
import sys
import time
from optparse import OptionParser

def main(input_file):
	"""Performs the postprocessing as specified in the provided xml file"""
	# Try to parse the supplied xml input file, exit if it fails.
	try:
		xml = XmlInterface(input_file)
	except:
		print("Error parsing XML file.")
		sys.exit()
	# Copy the postprocess settings to a convenient variable
	postprocess_settings = xml.postprocess()

	# Print out the input xml
	print("Input XML:")
	print(xml)
	print("")
	sys.stdout.flush()

	# Get the list of measurements to be made
	actions = postprocess_settings.keys()

	# Loop through the actions and call the relevant functions
	for action in actions:
		if action == "auto_correlation":
			pyQCD.postprocessing.main \
			  .auto_correlation(postprocess_settings[action])
		elif action == "correlator":
			pass
		elif action == "pair_potential":
			pyQCD.postprocessing.main \
			  .pair_potential(postprocess_settings[action])
		elif action == "lattice_spacing":
			pyQCD.postprocessing.main \
			  .lattice_spacing(postprocess_settings[action])

if __name__ == "__main__":
	# Parse the command line arguments
	parser = OptionParser()
	parser.add_option("-i", "--input", action = "store", type = "string",
					  dest = "input_file")
	options, args = parser.parse_args()
	main(options.input_file)
