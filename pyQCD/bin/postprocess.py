import core.lattice
import interfaces.xmlinput
import interfaces.lattice
import interfaces.measurements
import postprocessing.main
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
		xml = interfaces.xmlinput.XmlInterface(options.input_file)
	except:
		print("Error parsing XML file.")
		sys.exit()

	postprocess_settings = xml.postprocess()

	print("Input XML:")
	print(xml)
	print("")

	actions = postprocess_settings.keys()

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
