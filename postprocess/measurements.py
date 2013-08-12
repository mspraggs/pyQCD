import pylab as pl

def pair_potential(b, r):
	"""Calculates the quark pair potential as b[0] * r - b[1] / r + b[2]"""
	return b[0] * r + b[1] / r + b[2]

def potential_params(data):
	"""Fits the potential function to data and returns the three fitting
	parameters"""

	r = pl.arange(1, pl.size(data) + 1)
	b0 = pl.array([1.0, 1.0, 1.0])

	err_func = lambda b, r, y: y - pair_potential(b, r)
	
	b, result = optimize.leastq(err_func, b0, args = (r, data))

	if result != 1:
		print("Warning! Fit failed.")

	return b

def calculate_potential(wilson_loops):
	"""Calculate the pair potential from the average wilson loops provided"""
	potentials = pl.zeros(pl.size(wilson_loops, axis = 0))
	# The quark separations
	r = pl.arange(1, pl.size(wilson_loops, axis = 1) + 1)
	# The function we'll use to fit the Wilson loops
	f = lambda b, t, W: W - b[0] * pl.exp(-b[1] * t)

	for i in xrange(pl.size(wilson_loops, axis = 0)):
		params, result = optimize.leastsq(f, [1., 1.],
										  args = (n, wilson_loops[i]))
		potentials[i] = params[1]

	return potentials

def auto_correlation(plaquettes, t):
	"""Calculates the auto correlation function"""
	mean_plaquette = pl.mean(plaquettes)
	return pl.mean((plaquettes - mean_plaquette) \
				   * (pl.roll(plaquettes, -t) - mean_plaquette))

def calculate_spacing(wilson_loops):
	"""Calculates the lattice spacing using the Sommer scale"""
	fit_params = potential_params(wilson_loops)
	return 0.5 / pl.sqrt((1.65 + fit_params[1]) / fit_params[2])
