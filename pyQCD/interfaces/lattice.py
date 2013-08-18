import numpy as np
import itertools

class LatticeInterface:

	def __init__(self, lattice):
		"""Creates an interface for the supplied lattice object."""
		self.lattice = lattice

	def get_links(self):
		"""Extracts links from the lattice object as a list of numpy matrices.
		The returned list is flattened such that the index may be found from
		the four spacetime coordinates (t, x, y, z) and the link direction (mu)
		using the following formula:

		link_index = mu + 4 * z + 4 * L * y + 4 * L**2 * x + 4 * L**3 * t

		where L is the spatial extent of the lattice."""
		out = []
		r = xrange(self.lattice.L)
		t = xrange(self.lattice.T)
		links = itertools.product(t, r, r, r, range(4))
		
		for link in links:
			out.append(np.matrix(self.lattice.get_link(link)))
			
		return out

	def set_links(links):
		"""Sets the links of the lattice equal to those in the supplied list.
		The list should be a flattened array of 3x3 numpy matrices, with the
		flattening regime such that the list index is obtainable using the
		following formula:

		link_index = mu + 4 * z + 4 * L * y + 4 * L**2 * x + 4 * L**3 * t

		where t, x, y and z indicate the lattice site, mu the link direction
		and L the spatial extent of the lattice"""
		out = []
		r = xrange(self.lattice.L)
		t = xrange(self.lattice.T)
		link_coords = itertools.product(t, r, r, r, range(4))
		
		index = 0
		
		for link_coords in link_coords:
			temp_link = [[col for col in row] for row in links[i]]
			self.lattice.set_link(link)
			i += 1
			
	def get_wilson_loops(self, loop_config):
		"""Calculates the expectation values of all Wilson loops of size nxm,
		with n = 1, 2, ... , t_max and m = 1, 2, ... , r_max. The function
		will return a numpy array of size (r_max - 1)x(t_max - 1), with each
		column representing a different time and each row representing a
		different spatial separation. The gauge field is smeared using stout
		smearing, as defined by the num_field_smears and the
		field_smearing_param variable."""
		out = np.zeros((loop_config['r_max'] - 1,
						loop_config['t_max'] - 1))
		
		for r in xrange(1, loop_config['r_max']):
			for t in xrange(1, loop_config['t_max']):
				out[r - 1, t - 1] \
				  += self.lattice \
				  .av_wilson_loop(r, t, loop_config['num_field_smears'],
								  loop_config['field_smearing_param'],)
				
		return out

	def get_propagator(self, prop_config):
		"""Extracts the Wilson fermion propagator, as calculated for the
		specified mass, lattice spacing and source site, as a flattened list of
		numpy matrices. The list index corresponds to the lattice coordinates t,
		x, y and z via the following formula:

		site_index = z + L * y + L**2 * x + L**3 * t

		where L is the spatial extent of the lattice.

		It is possible to apply stout smearing to the lattice gauge field
		and Jacobi smearing to the propagator source and sink using the
		given function arguments."""
		raw_propagator = self \
		  .lattice.propagator(prop_config['mass'],
							  prop_config['a'],
							  prop_config['source_site'],
							  prop_config['num_field_smears'],
							  prop_config['field_smearing_param'],
							  prop_config['num_source_smears'],
							  prop_config['source_smearing_param'],
							  prop_config['num_sink_smears'],
							  prop_config['sink_smearing_param'],
							  prop_config['solver_method'])

		return [np.matrix(matrix) for matrix in raw_propagator]
