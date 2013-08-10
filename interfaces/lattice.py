import numpy as np
import itertools

def get_links(lattice):
	"""Extracts links from lattice as a compound list of numpy arrays"""
	out = []
	r = xrange(lattice.n_points)
	links = intertools.product(r, r, r, r, range(4))

	for link in links:
		out.append(np.matrix(lattice.get_link(link)))

	return out

def set_links(lattice, links):
	"""Inserts links into lattice"""
	out = []
	r = xrange(lattice.n_points)
	link_coords = intertools.product(r, r, r, r, range(4))

	index = 0
	
	for link_coords in link_coords:
		temp_link = [[col for col in row] for row in links[i]]
		lattice.set_link(link)
		i += 1

	return out

def get_wilson_loops(lattice, rmax, tmax, n_smears = 0):
    """Calculates a series of Wilson loops up to the maximum r and t values"""
    out = np.zeros((rmax - 1, tmax - 1))
    
    for r in xrange(1, rmax):
        for t in xrange(1, tmax):
            out[r - 1, t - 1] += lattice.av_wilson_loop(r, t, n_smears)
            
    return out

def get_propagator(lattice, mass, spacing = 1.0, source = [0,0,0,0],
				   n_smears = 0, n_src_smears = 0, src_param = 1.0,
				   n_sink_smears = 0, sink_param = 1.0, solver_method = 0):
	"""Extracts the propagator as a list of matrices"""
	raw_propagator = lattice.propagator(mass, spacing, source, n_smears,
										n_src_smears, src_param,
										n_sink_smears, sink_param, solver_method)

	return [np.matrix(matrix) for matrix in raw_propagator]
