import xml.etree.ElementTree as ET
import dicts
import os.path as op

class XmlInterface:

	def __init__(self, filename):
		"""Creates an interface to the input xml file that defines the
		various settings used in the simulation or postprocessing script.
		The default settings, where they exist, are specified in the default
		xml file, default."""
		self.filename = filename
		
		xmltree = ET.parse(filename)
		xmlroot = xmltree.getroot()

		self.settings = self.parse_tree(xmlroot)

		self.fill_defaults(self.settings, dicts.defaults)
		self.fill_dicts(self.settings)

	def __str__(self):
		"""Returns the contents of the xml file as a string."""

		return = open(self.filename).read()

	def fill_defaults(self, settings, defaults):
		"""Loops through the settings dictionary, looks for corresponding keys
		in the defaults dictionary and applies the values there, if any, in
		a recursive fashion."""

		keys = defaults.keys()

		do_not_adds = ["measurements",
					   "propagator",
					   "plaquette",
					   "wilson_loop",
					   "postprocess",
					   "auto_correlation",
					   "lattice_spacing",
					   "input",
					   "store",
					   "plot",
					   "lattice",
					   "simulation",
					   "gauge_action",
					   "pair_potential"]

		for key in keys:
			if settings.has_key(key):
				if settings[key] == None:
					if defaults[key] == None:
						raise ET. \
						  ParseError("Missing required configuration tag.")
					else:
						settings[key] = defaults[key]

				elif key == "input":
					for input_dict in settings[key]:
						self.fill_defaults(input_dict, defaults[key][0])
				elif type(settings[key]) == dict:
					self.fill_defaults(settings[key], defaults[key])
			else:
				if do_not_adds.count(key) == 0:
					if not defaults.has_key(key):
						raise ET \
						  .ParseError("Missing required configuration tag.")
					else:
						settings.update({key: defaults[key]})
						if type(defaults[key]) == dict:
							self.fill_defaults(settings[key], defaults[key])

	def fill_dicts(self, settings):
		"""Loops through the settings dictionary and, if possible, uses the
		dictionaries in dicts to swap the relevant keys for the values in
		those dicitionaries."""

		keys = settings.keys()

		for key in keys:
			if key == "input":
				for input_dict in settings[key]:
					self.fill_dicts(input_dict)
			elif type(settings[key]) == str:
				for dictionary in dicts.dicts:
					if dictionary.has_key(settings[key]):
						settings[key] = dictionary[settings[key]]
						break
			elif type(settings[key]) == dict:
				self.fill_dicts(settings[key])
		
	def parse_tree(self, root):
		"""Loop through the xml elements under the root element and add the
		tags and contents to a dictionary. For each element that is parent
		to other elements, the function is called recursively."""

		output = []
		inputs = []
		inputs_exist = False
		children = list(root)

		for child in children:
			if len(list(child)) == 0:
				try:
					output.append((child.tag, eval(child.text)))
				except (NameError, TypeError):
					output.append((child.tag, child.text))
			else:
				if child.tag == "input":
					inputs.append(self.parse_tree(child))
					inputs_exist = True
				else:
					output.append((child.tag, self.parse_tree(child)))

		if inputs_exist: output.append(("input", inputs))

		return dict(output)

	def gauge_action(self):
		"""Return the gauge action settings dictionary."""
		return self.settings["gauge_action"]

	def lattice(self):
		"""Return the lattice settings dictionary."""
		return self.settings["lattice"]

	def simulation(self):
		"""Return the simulation settings dictionary."""
		return self.settings["simulation"]
		
	def measurements(self):
		"""Return measurement settings, if any, as a dictionary."""
		try:
			return self.settings["measurements"]
		except KeyError:
			return dict()

	def postprocess(self):
		"""Return postprocess settings, if any, as a dictionary."""
		try:
			return self.settings["postprocess"]
		except KeyError:
			return dict()
