import xml.etree.ElementTree as ET
import IPython
		
update_methods = {'HEATBATH': 0,
				  'STAPLE_MONTE_CARLO': 1,
				  'MONTE_CARLO': 2}

gauge_actions = {'WILSON': 0,
				 'RECTANGLE_IMPROVED': 1,
				 'TWISTED_RECTANGLE_IMPROVED': 2}

solver_methods = {'BiCGSTAB': 0,
				  'ConjugateGradient': 1}

truefalse = {'TRUE': 1,
			 'FALSE': 0}

def validate_tags(root, tags):
	"""Checks whether the supplied tags are valid or not"""
	out = []
	for tag in tags:
		element = root.findall(tag)
		if element == None or len(element) != 1:
			return None
		else:
			out.append(element[0])
	return out

def convert_content(element, return_type):
	"""Extracts the data within the element and converts it to the
	specified type"""

	content = element.text

	if return_type == int:
		return int(eval(content))
	elif return_type == float:
		return float(eval(content))
	elif return_type == list:
		return eval(content)
	elif return_type == str:
		return content
	elif type(return_type) == dict:
		return return_type[content]
	else:
		return content

def extract_defaults(root, tags, defaults, return_types):
	"""Looks for the tag in root and returns default if it isn't found"""
	elements = validate_tags(root, tags)
	if elements == None:
		return zip(tags, defaults)
	else:
		out = []
		for i in xrange(len(elements)):
			out.append((tags[i], convert_content(elements[i], return_types[i])))
		return out

def extract(root, tags, return_types):
	"""Extract the data from the specified tags"""
	elements = validate_tags(root, tags)
	if elements == None:
		return None
	else:
		out = []
		for i in xrange(len(elements)):
			out.append((tags[i], convert_content(elements[i], return_types[i])))
		return out

class Xml:

	def __init__(self, fname):
		"""Object constructor - accepts xml filename"""
		self.filename = fname
		self.xmltree = ET.parse(self.filename)

	def check_root(self):
		"""Checks to see that the root tag is correct"""
		root = self.xmltree.getroot()
		if root.tag != 'pyQCD':
			return False
		else:
			return True
		
	def parse_lattice(self):
		"""Extracts lattice parameters from the xml"""
		lattice = validate_tags(self.xmltree, ["lattice"])

		if lattice == None:
			return None
		else:
			lattice_settings = []
			tags = ["T", "L"]
			types = [int, int]
			temp_list = extract(lattice[0], tags, types)

			if temp_list == None:
				return None
			else:
				lattice_settings += temp_list

				return dict(lattice_settings)

	def parse_parallel_update(self, root):
		"""Checks for the parallel update settings"""
		parallel_update = validate_tags(root, ["parallel_update"])

		out = []
		
		if parallel_update == None:
			return dict([("enabled", 0), ("block_size", 1)])
		else:
			optional_tags = ["enabled", "block_size"]
			optional_types = [truefalse, int]
			optional_defaults = [0, 1]

			optional_list = extract_defaults(parallel_update[0], optional_tags,
											 optional_defaults, optional_types)

			return dict(optional_list)

	def parse_timing_run(self, root):
		"""Checks for timing run settings"""
		timing_run = validate_tags(root, ["timing_run"])

		out = []

		if timing_run == None:
			return dict([("enabled", 0), ("num_configurations", 0)])
		else:
			optional_tags = ["enabled", "num_configurations"]
			optional_types = [truefalse, int]
			optional_defaults = [0, 0]

			optional_list = extract_defaults(timing_run[0], optional_tags,
											 optional_defaults, optional_types)

			return dict(optional_list)
		
	def parse_simulation(self):
		"""Extracts simulation parameters from the xml"""
		simulation = validate_tags(self.xmltree, ["simulation"])

		if simulation == None:
			return None
		else:
			simulation_settings = []
			required_tags = ["num_configurations", "measurement_spacing"]
			optional_tags = ["update_method"]
			required_types = [int, int]
			optional_types = [update_methods]
			optional_defaults = [0]

			required_list = extract(simulation[0], required_tags, required_types)
			optional_list = extract_defaults(simulation[0], optional_tags,
											 optional_defaults, optional_types)

			if required_list == None:
				return None
			else:
				simulation_settings += required_list

			simulation_settings += optional_list

			simulation_settings \
			  .append(("parallel_update",
					   self.parse_parallel_update(simulation[0])))
			simulation_settings \
			  .append(("timing_run", self.parse_timing_run(simulation[0])))

		return dict(simulation_settings)

	def parse_gauge_action(self):
		"""Extracts information about the gauge action"""
		gauge_action = validate_tags(self.xmltree, ["gauge_action"])

		if gauge_action == None:
			return None
		else:
			gauge_action_settings = []
			required_tags = ["type", "beta"]
			optional_tags = ["u0"]
			required_types = [gauge_actions, float]
			optional_types = [float]
			optional_defaults = [1.0]

			required_list = extract(gauge_action[0], required_tags,
									required_types)
			optional_list = extract_defaults(gauge_action[0], optional_tags,
											 optional_defaults, optional_types)

			if required_list == None:
				return None
			else:
				gauge_action_settings += required_list
				gauge_action_settings += optional_list

				return dict(gauge_action_settings)

	def parse_plaquette(self, root):
		"""Extracts plaquette measurement parameters from the xml"""
		
		plaquette = validate_tags(root, ["plaquette"])

		if plaquette == None:
			return None
		else:
			plaquette_settings = []
			required_tags = ["filename"]
			required_types = [str]

			required_list = extract(plaquette[0], required_tags,
									required_types)

			if required_list == None:
				return None
			else:
				plaquette_settings += required_list
				return dict(plaquette_settings)

	def parse_wilson_loop(self, root):
		"""Extracts wilson loop measurement parameters from the xml"""		
		wilson_loop = validate_tags(root, ["wilson_loop"])

		if wilson_loop == None:
			return None
		else:
			wilson_loop_settings = []
			required_tags = ["filename", "r_max", "t_max"]
			optional_tags = ["num_field_smears", "field_smearing_param"]
			required_types = [str, int, int]
			optional_types = [int, float]
			optional_defaults = [0, 1.0]

			required_list = extract(wilson_loop[0], required_tags,
									required_types)
			optional_list = extract_defaults(wilson_loop[0], optional_tags,
											 optional_defaults, optional_types)

			if required_list == None:
				return None
			else:
				wilson_loop_settings += required_list
				wilson_loop_settings += optional_list

				return dict(wilson_loop_settings)

	def parse_propagator(self, root):
		"""Extracts wilson loop measurement parameters from the xml"""		
		propagator = validate_tags(root, ["propagator"])

		if propagator == None:
			return None
		else:
			propagator_settings = []
			required_tags = ["filename", "mass"]
			optional_tags = [["a"],
							 ["source_site"],
							 ["solver_method"],
							 ["num_source_smears", "source_smearing_param"],
							 ["num_sink_smears", "sink_smearing_param"],
							 ["num_field_smears", "field_smearing_param"]]
			required_types = [str, float]
			optional_types = [[float],
							  [list],
							  [str],
							  [int, float],
							  [int, float],
							  [int, float]]
			optional_defaults = [[1.0],
								 [[0,0,0,0]],
								 [0],
								 [0, 1.0],
								 [0, 1.0],
								 [0, 1.0]]

			required_list = extract(propagator[0], required_tags,
									required_types)
			optional_list = []

			for i in xrange(len(optional_tags)):
				optional_list += extract_defaults(propagator[0],
												  optional_tags[i],
												  optional_defaults[i],
												  optional_types[i])

			if required_list == None:
				return None
			else:
				propagator_settings += required_list
				propagator_settings += optional_list

				return dict(propagator_settings)

	def parse_measurements(self):
		"""Parses all the relevant measurement code"""
		measurements = validate_tags(self.xmltree, ["measurements"])

		if measurements == None:
			return None
		else:
			measurement_settings = []

			plaquette_settings = self.parse_plaquette(measurements[0])

			if plaquette_settings != None:
				measurement_settings.append(("plaquette", plaquette_settings))

			wilson_loop_settings = self.parse_wilson_loop(measurements[0])

			if wilson_loop_settings != None:
				measurement_settings.append(("wilson_loop",
											 wilson_loop_settings))

			propagator_settings = self.parse_propagator(measurements[0])

			if propagator_settings != None:
				measurement_settings.append(("propagator", propagator_settings))


			return dict(measurement_settings)
