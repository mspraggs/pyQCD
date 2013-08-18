import pylab as pl
from scipy import optimize
import IPython

def pair_potential(b, r):
	"""Calculates the quark pair potential as b[0] * r - b[1] / r + b[2]"""
	return b[0] * r + b[1] / r + b[2]

def potential_params(data):
	"""Fits the potential function to data and returns the three fitting
	parameters"""
	if len(pl.shape(data)) == 1:
	
		r = pl.arange(1, pl.size(data) + 1)
		b0 = pl.array([1.0, 1.0, 1.0])

		err_func = lambda b, r, y: y - pair_potential(b, r)
		b, result = optimize.leastsq(err_func, b0, args = (r, data))

		if result != 1:
			print("Warning! Fit failed.")

		return b

	else:
		b_store = pl.zeros((pl.size(data, axis = 0), 3))

		for i in xrange(pl.size(data, axis = 0)):
			b_store[i] = potential_params(data[i])

		return b_store

def calculate_potential(wilson_loops):
	"""Calculate the pair potential from the average wilson loops provided"""
	if len(pl.shape(wilson_loops)) == 2:
		potentials = pl.zeros(pl.size(wilson_loops, axis = 0))
		# The quark separations
		r = pl.arange(1, pl.size(wilson_loops, axis = 0) + 1)
		t = pl.arange(1, pl.size(wilson_loops, axis = 1) + 1)
		# The function we'll use to fit the Wilson loops
		f = lambda b, t, W: W - b[0] * pl.exp(-b[1] * t)
		
		for i in xrange(pl.size(wilson_loops, axis = 0)):
			params, result = optimize.leastsq(f, [1., 1.],
											  args = (t, wilson_loops[i]))
			potentials[i] = params[1]

	else:
		potentials = pl.zeros(pl.shape(wilson_loops)[:-1])

		for i in xrange(pl.size(wilson_loops, axis = 0)):
			potentials[i] = calculate_potential(wilson_loops[i])

	return potentials

def auto_correlation(plaquettes):
	"""Calculates the auto correlation function"""
	mean_plaquette = pl.mean(plaquettes)

	num_configs = pl.size(plaquettes, axis = 0) / 2

	auto_corr = pl.zeros(num_configs)

	for t in xrange(num_configs):
		auto_corr[t] = pl.mean((plaquettes - mean_plaquette) \
							   * (pl.roll(plaquettes, -t) - mean_plaquette))
	
	return auto_corr

def calculate_spacing(wilson_loops):
	"""Calculates the lattice spacing using the Sommer scale"""
	fit_params = potential_params(wilson_loops)
	if len(pl.shape(fit_params)) > 1:
		return 0.5 / pl.sqrt((1.65 + fit_params[:,1]) / fit_params[:,0])
	else:
		return 0.5 / pl.sqrt((1.65 + fit_params[1]) / fit_params[0])		
