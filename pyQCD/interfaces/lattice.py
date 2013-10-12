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
			
		return [np.matrix(self.lattice.get_link(list(link)))
				for link in links]

	def set_links(self, links):
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
		
		for link,link_coord in zip(links,link_coords):
			self.lattice.set_link(list(link_coord), link.tolist())
			
	def get_wilson_loops(self, r_max, t_max, num_field_smears = 0,
						 field_smearing_param = 1.0):
		"""Calculates the expectation values of all Wilson loops of size nxm,
		with n = 1, 2, ... , t_max and m = 1, 2, ... , r_max. The function
		will return a numpy array of size (r_max - 1)x(t_max - 1), with each
		column representing a different time and each row representing a
		different spatial separation. The gauge field is smeared using stout
		smearing, as defined by the num_field_smears and the
		field_smearing_param variable."""
		out = np.zeros((r_max - 1, t_max- 1))
		
		for r in xrange(1, r_max):
			for t in xrange(1, t_max):
				out[r - 1, t - 1] \
				  += self.lattice \
				  .av_wilson_loop(r, t, num_field_smears, field_smearing_param)
				
		return out

	def get_propagator(self, mass,
					   a = 1.0,
					   source_site = [0, 0, 0, 0],
					   num_field_smears = 0,
					   field_smearing_param = 1.0,
					   num_source_smears = 0,
					   source_smearing_param = 1.0,
					   num_sink_smears = 0,
					   sink_smearing_param = 1.0,
					   solver_method = 0,
					   verbosity = 0):
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
		  .lattice.propagator(mass,
							  a,
							  source_site,
							  num_field_smears,
							  field_smearing_param,
							  num_source_smears,
							  source_smearing_param,
							  num_sink_smears,
							  sink_smearing_param,
							  solver_method,
							  verbosity)

		return [np.matrix(matrix) for matrix in raw_propagator]
