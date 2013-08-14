import core.lattice
import interfaces.xmlinput
import interfaces.lattice
import interfaces.measurements
import postprocessing.xmlroutines
import sys
import time
from optparse import OptionParser

def main():
	"""Main execution function"""
	parser = OptionParser()
	parser.add_option("-i", "--input", action = "store", type = "string",
					  dest = "input_file")

	options, args = parser.parse_args()

	try:
		xmlinput = interfaces.xmlinput.XmlInterface(options.input_file)
	except:
		print("Error parsing XML file.")
		sys.exit()

	postprocess_settings = xmlinput.postprocess()

	print("Input XML:")
	print(xmlinput)
	print("")

	actions = postprocess_settings.keys()

	for action in actions:
		if action == "auto_correlation":

		elif action == "correlator"

		elif action == "pair_potential"
