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
			print("Warning! Returning None...")
			return None
		else:
			lattice_settings = []
			tags = ["T", "L"]
			elements = validate_tags(lattice[0], tags)

			if elements == None:
				print("Warning! Returning None...")
				return None
			else:
				lattice_settings.append((tags[0], int(eval(elements[0].text))))
				lattice_settings.append((tags[1], int(eval(elements[1].text))))

				return dict(lattice_settings)

	def parse_simulation(self):
		"""Extracts simulation parameters from the xml"""
		simulation = validate_tags(self.xmltree, ["simulation"])

		if simulation == None:
			return None
		else:
			simulation_settings = []
			tags = ["num_configurations", "measurement_spacing",
					"update_method", "parallel_update", "timing_run"]
			required_elements = validate_tags(simulation[0], tags[0:2])

			if required_elements == None:
				return None
			else:
				simulation_settings \
				  .append((tags[0], int(eval(required_elements[0].text))))
				simulation_settings \
				  .append((tags[1], int(eval(required_elements[1].text))))

			update_method = validate_tags(simulation[0], tags[2:3])

			if update_method == None:
				simulation_settings.append((tags[2], 0))
			else:
				simulation_settings \
				  .append((tags[2], update_methods[update_method[0].text]))

			parallel_update = validate_tags(simulation[0], tags[3:4])

			if parallel_update == None:
				simulation_settings.append(("parallel_flag", 0))
				simulation_settings.append(("block_size", 1))
			else:
				parallel_tags = ["enabled", "block_size"]
				parallel_elements = validate_tags(parallel_update[0],
												  parallel_tags)

				if parallel_elements == None:
					return None
				else:
					parallel_flag \
					  = 1 if parallel_elements[0].text == 'TRUE' else 0
					simulation_settings.append(("parallel_flag", parallel_flag))

					simulation_settings \
					  .append((parallel_tags[1],
							   int(eval(parallel_elements[1].text))))

			timing_run = validate_tags(simulation[0], tags[4:5])

			if timing_run == None:
				simulation_settings.append(("timing_flag", 0))
				simulation_settings.append(("num_timing_configs", 0))
			else:
				timing_tags = ["enabled", "num_configurations"]
				timing_elements = validate_tags(timing_run[0], timing_tags)

				if timing_elements == None:
					return None
				else:
					timing_flag \
					  = 1 if timing_elements[0].text == 'TRUE' else 0
					simulation_settings.append(("timing_flag", timing_flag))

					simulation_settings \
					  .append(("num_timing_configs",
							   int(eval(timing_elements[1].text))))

					return dict(simulation_settings)

	def parse_gauge_action(self):
		"""Extracts information about the gauge action"""
		gauge_action = validate_tags(self.xmltree, ["gauge_action"])

		if gauge_action == None:
			return None
		else:
			gauge_action_settings = []
			tags = ["type", "beta", "u0"]
			elements = validate_tags(gauge_action[0], tags[0:2])

			if elements == None:
				return None
			else:
				gauge_action_settings.append((tags[0],
											  gauge_actions[elements[0].text]))
				gauge_action_settings.append((tags[1], eval(elements[1].text)))

				improvement_element = validate_tags(gauge_action[0], tags[2:3])

				if improvement_element == None:
					gauge_action_settings.append((tags[2], 1.0))
				else:
					gauge_action_settings \
					  .append((tags[2], eval(improvement_element[0].text)))

					return dict(gauge_action_settings)

	def parse_plaquette(self):
		"""Extracts plaquette measurement parameters from the xml"""
		measurements = validate_tags(self.xmltree, ["measurements"])
		if measurements == None:
			return None
		
		plaquette = validate_tags(measurements[0], ["plaquette"])

		if plaquette == None:
			return None
		else:
			plaquette_settings = []
			tags = ["filename"]
			elements = validate_tags(plaquette[0], tags)

			if elements == None:
				return None
			else:
				plaquette_settings.append((tags[0], elements[0].text))

				return dict(plaquette_settings)

	def parse_wilson_loop(self):
		"""Extracts wilson loop measurement parameters from the xml"""
		measurements = validate_tags(self.xmltree, ["measurements"])
		if measurements == None:
			return None
		
		wilson_loop = validate_tags(measurements[0], ["wilson_loop"])

		if wilson_loop == None:
			return None
		else:
			wilson_loop_settings = []
			tags = ["filename", "r_max", "t_max", "num_field_smears",
					"field_smearing_param"]
			elements = validate_tags(wilson_loop[0], tags[0:3])

			if elements == None:
				return None
			else:
				wilson_loop_settings.append((tags[0], elements[0].text))					  
				wilson_loop_settings \
				  .append((tags[1], int(eval(elements[1].text))))
				wilson_loop_settings \
				  .append((tags[2], int(eval(elements[2].text))))

				link_smear_elements = validate_tags(wilson_loop[0], tags[3:5])

				if link_smear_elements == None:
					wilson_loop_settings.append((tags[3], 0))
					wilson_loop_settings.append((tags[4], 1.0))
				else:					  
					wilson_loop_settings \
					  .append((tags[3], int(eval(link_smear_elements[0].text))))
					wilson_loop_settings \
					  .append((tags[4], eval(link_smear_elements[1].text)))

				return dict(wilson_loop_settings)

	def parse_propagator(self):
		"""Extracts wilson loop measurement parameters from the xml"""
		measurements = validate_tags(self.xmltree, ["measurements"])
		if measurements == None:
			return None
		
		propagator = validate_tags(measurements[0], ["propagator"])

		if propagator == None:
			return None
		else:
			propagator_settings = []
			tags = ["filename", "mass", "a", "source_site", "solver_method",
					"num_source_smears", "source_smearing_param",
					"num_sink_smears", "sink_smearing_param", "num_field_smears",
					"field_smearing_param"]
			elements = validate_tags(propagator[0], tags[0:2])

			if elements == None:
				return None
			else:
				propagator_settings.append((tags[0], elements[0].text))
				propagator_settings \
				  .append((tags[1], eval(elements[1].text)))

				spacing_element = validate_tags(propagator[0], tags[2:3])

				if spacing_element == None:
					propagator_settings.append((tags[2], 1.0))
				else:					
					propagator_settings \
					  .append((tags[2], eval(spacing_element[0].text)))

				source_element = validate_tags(propagator[0], tags[3:4])

				if spacing_element == None:
					propagator_settings.append((tags[3], [0, 0, 0, 0]))
				else:
					propagator_settings \
					  .append((tags[3], eval(source_element[0].text)))

				method_element = validate_tags(propagator[0], tags[4:5])

				if method_element == None:
					propagator_settings.append((tags[4], 0))
				else:
					propagator_settings \
					  .append((tags[4], solver_methods[method_element[0].text]))

				source_smear_elements = validate_tags(propagator[0], tags[5:7])

				if source_smear_elements == None:
					propagator_settings.append((tags[5], 0))
					propagator_settings.append((tags[6], 1.0))
				else:
					propagator_settings \
					  .append((tags[5],
							   int(eval(source_smear_elements[0].text))))
					propagator_settings \
					  .append((tags[6], eval(source_smear_elements[1].text)))

				sink_smear_elements = validate_tags(propagator[0], tags[7:9])

				if sink_smear_elements == None:
					propagator_settings.append((tags[7], 0))
					propagator_settings.append((tags[8], 1.0))
				else:				  
					propagator_settings \
					  .append((tags[7], int(eval(sink_smear_elements[0].text))))
					propagator_settings \
					  .append((tags[8], eval(sink_smear_elements[1].text)))

				link_smear_elements = validate_tags(propagator[0], tags[9:11])

				if link_smear_elements == None:
					propagator_settings.append((tags[9], 0))
					propagator_settings.append((tags[10], 1.0))
				else:					  
					propagator_settings \
					  .append((tags[9], int(eval(link_smear_elements[0].text))))
					propagator_settings \
					  .append((tags[10], eval(link_smear_elements[1].text)))

				return dict(propagator_settings)

	def parse_measurements(self):
		"""Parses all the relevant measurement code"""
		measurements = validate_tags(self.xmltree, ["measurements"])

		if measurements == None:
			return None
		else:
			measurement_settings = []

			plaquette_settings = self.parse_plaquette()

			if plaquette_settings != None:
				measurement_settings.append(("plaquette", plaquette_settings))

			wilson_loop_settings = self.parse_wilson_loop()

			if wilson_loop_settings != None:
				measurement_settings.append(("wilson_loop",
											 wilson_loop_settings))

			propagator_settings = self.parse_propagator()

			if propagator_settings != None:
				measurement_settings.append(("propagator", propagator_settings))


			return dict(measurement_settings)
