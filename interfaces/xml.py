import xml.etree.ElementTree as ET

def validate_tags(root, tags):
	"""Checks whether the supplied tags are valid or not"""
	out = []
	for tag in tags:
		element = root.findall(tag)
		if element == None or len(element) > 1:
			return None
		else:
			out.append(element)
	return out

class Xml:

	self.update_methods = {'HEATBATH': 0,
						   'STAPLE_MONTE_CARLO': 1,
						   'MONTE_CARLO': 2}

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
		lattice = validate_tags(xmltree, ["lattice"])

		if lattice == None:
			return None
		else:
			lattice_settings = []
			tags = ["T", "L", "a"]
			elements = validate_tags(lattice, tags)

			if elements == None:
				return None
			else:
				lattice_settings.append((tags[0], int(eval(elements[0].text))))
				lattice_settings.append((tags[1], int(eval(elements[1].text))))
				lattice_settings.append((tags[2], eval(elements[2].text)))

				return dict(lattice_settings)

	def parse_lattice(self):
		lattice = validate_tags(self.xmltree, ["lattice"])

		if lattice == None:
			return None
		else:
			lattice_settings = []
			tags = ["T", "L", "a"]
			elements = validate_tags(lattice, tags)

			if elements == None:
				return None
			else:
				lattice_settings.append((tags[0], int(eval(elements[0].text))))
				lattice_settings.append((tags[1], int(eval(elements[1].text))))
				lattice_settings.append((tags[2], eval(elements[2].text)))

				return dict(lattice_settings)

	def parse_simulation(self):
		simulation = validate_tags(self.xmltree, ["simualation"])

		if simulation == None:
			return None
		else:
			simulation_settings = []
			tags = ["num_configurations", "measurement_spacing",
					"update_method", "parallel_update"]
			required_elements = validate_tags(simulation, tags[0:2])

			if required_elements == None:
				return None
			else:
				simulation_settings \
				  .append((tags[0], int(eval(required_elements[0].text))))
				simulation_settings \
				  .append((tags[1], int(eval(required_elements[1].text))))

			update_method = validate_tags(simulation, tags[2:3])

			if update_method == None:
				simulation_settings.append((tags[2], 0))
			else:
				simulation_settings \
				  .append((tags[2], self.update_methods[update_method[0].text]))

			parallel_update = validate_tags(simulation, tags[3:4])

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

		return dict(simulation_settings)
