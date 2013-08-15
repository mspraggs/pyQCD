import numpy as np
import itertools

class LatticeInterface:

	def __init__(self, lattice):
		"""Constructor"""
		self.lattice = lattice

	def get_links(self):
		"""Extracts links from lattice as a compound list of numpy arrays"""
		out = []
		r = xrange(lattice.L)
		t = xrange(lattice.T)
		links = intertools.product(t, r, r, r, range(4))
		
		for link in links:
			out.append(np.matrix(self.lattice.get_link(link)))
			
		return out

	def set_links(links):
		"""Inserts links into lattice"""
		out = []
		r = xrange(lattice.L)
		t = xrange(lattice.T)
		link_coords = intertools.product(t, r, r, r, range(4))
		
		index = 0
		
		for link_coords in link_coords:
			temp_link = [[col for col in row] for row in links[i]]
			self.lattice.set_link(link)
			i += 1
			
	def get_wilson_loops(self, loop_config):
		"""Calculates a series of Wilson loops up to the maximum r and t
		values"""
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
		"""Extracts the propagator as a list of matrices"""
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
