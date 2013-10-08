import pylab as pl

def bin(X, binsize = 1):
	"""Split X into bins and return the average of each bin as a new set of
	measurements."""
	if binsize == 1:
		return X;
	else:
		extra = 0 if pl.size(X, axis = 0) % binsize == 0 else 1
		dims = [i for i in pl.shape(X)]
		dims[0] = dims[0] / binsize + extra
		dims = tuple(dims)
		X_binned = pl.zeros(dims)

		for i in xrange(pl.size(X_binned, axis = 0)):
			X_binned[i] = pl.mean(X[i * binsize:(i + 1) * binsize], axis = 0)

		return X_binned

def bootstrap(X):
	"""Performs a bootstrap resampling of X."""
	return X[pl.randint(0, pl.size(X, axis = 0), pl.size(X, axis = 0))]

def bootstrap_measurement(X, func, num_bootstraps, binsize):
	"""Calculates bootstrap statistics for a given measurement function."""
	out = []

	if binsize > 0:
		X = bin(X, binsize)

	if num_bootstraps > 0:
		for i in xrange(num_bootstraps):
			X_bootstrap = bootstrap(X)
			out.append(pl.mean(func(X_bootstrap), axis=0))
	else:
		out.append(func(X))

	out = pl.array(out) if len(out) > 1 else pl.array(out[0])

	out_mean = pl.mean(out, axis=0)
	out_std = pl.std(out, axis=0)

	return out_mean, out_std
