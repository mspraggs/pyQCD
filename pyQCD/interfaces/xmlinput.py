import xml.etree.ElementTree as ET
import dicts
import os.path as op
import IPython

class XmlInterface:

	def __init__(self, filename, default = "simulation_default.xml"):
		"""Constructor"""
		self.filename = filename
		
		xmltree = ET.parse(filename)
		xmlroot = xmltree.getroot()

		directory, currfile = op.split(op.realpath(__file__))
		default = op.join(directory, default)
		defaulttree = ET.parse(default)
		defaultroot = defaulttree.getroot()

		self.settings = self.parse_tree(xmlroot)
		self.defaults = self.parse_tree(defaultroot)

		self.fill_defaults(self.settings, self.defaults)
		self.fill_dicts(self.settings)

	def __str__(self):
		"""Returns file as string"""

		string_out = ""
		lines = open(self.filename).readlines()

		for line in lines:
			string_out += line

		return string_out

	def fill_defaults(self, settings, defaults):
		"""Loops through settings and applied defaults"""

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
		"""Loops through the settings and applies the dictionaries"""

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
		"""Iterate through the tree and add the contents to a list"""

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
		"""Return the gauge action settings"""
		return self.settings["gauge_action"]

	def lattice(self):
		"""Return the lattice settings"""
		return self.settings["lattice"]

	def simulation(self):
		"""Return the simulation settings"""
		return self.settings["simulation"]
		
	def measurements(self):
		"""Return measurement settings, if any"""
		try:
			return self.settings["measurements"]
		except KeyError:
			return dict()

	def postprocess(self):
		"""Return postprocess settings, if any"""
		try:
			return self.settings["postprocess"]
		except KeyError:
			return dict()
