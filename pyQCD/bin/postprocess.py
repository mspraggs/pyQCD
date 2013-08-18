import core.lattice
import interfaces.xmlinput
import interfaces.lattice
import interfaces.measurements
import postprocessing.main
import sys
import time
from optparse import OptionParser

def main():
	"""Main execution function for postprocessing"""
	# Parse the command line arguments
	parser = OptionParser()
	parser.add_option("-i", "--input", action = "store", type = "string",
					  dest = "input_file")
	options, args = parser.parse_args()

	# Try to parse the supplied xml input file, exit if it fails.
	try:
		xml = interfaces.xmlinput.XmlInterface(options.input_file)
	except:
		print("Error parsing XML file.")
		sys.exit()
	# Copy the postprocess settings to a convenient variable
	postprocess_settings = xml.postprocess()

	# Print out the input xml
	print("Input XML:")
	print(xml)
	print("")

	# Get the list of measurements to be made
	actions = postprocess_settings.keys()

	# Loop through the actions and call the relevant functions
	for action in actions:
		if action == "auto_correlation":
			postprocessing.main \
			  .auto_correlation(postprocess_settings[action])
		elif action == "correlator":
			pass
		elif action == "pair_potential":
			postprocessing.main \
			  .pair_potential(postprocess_settings[action])
		elif action == "lattice_spacing":
			postprocessing.main \
			  .lattice_spacing(postprocess_settings[action])

if __name__ == "__main__":
	main()
