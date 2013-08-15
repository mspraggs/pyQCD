import pylab as pl
import fileio
from numpy import save, load
import statistics
import measurements

def auto_correlation(settings):
	"""Calculate the autocorrelation"""

	inputs = settings["input"]

	output = []

	for i in inputs:
		if i["type"] != 0:
			print("Error, invalid input data. Skipping.")
		else:
			input_data = load(i["filename"])
			measurement = statistics \
			  .bootstrap_measurement(input_data,
									 measurements.auto_correlation,
									 settings["num_bootstraps"],
									 settings["bin_size"])

			t = pl.arange(pl.size(measurement[0], axis = 0))
			if settings.has_key("plot"):
				linestyle = i["linestyle"]["colour"] + i["linestyle"]["style"]
				pl.plot(t, measurement[0], linestyle, label = i["label"])

			if settings.has_key("store"):
				output.append(t)
				output.append(measurement[0])
				output.append(measurement[1])

	if settings.has_key("plot"):
		pl.show()

	if settings.has_key("store"):
		save(settings["store"]["filename"], output)

def pair_potential(settings):
	"""Calculate the pair potential"""

	inputs = settings["input"]

	output = []

	for i in inputs:
		if i["type"] != 1:
			print("Error, invalid input data. Skipping.")
		else:
			input_data = load(i["filename"])
			measurement = statistics \
			  .bootstrap_measurement(input_data,
									 measurements.calculate_potential,
									 settings["num_bootstraps"],
									 settings["bin_size"])

			r = pl.arange(1, pl.size(measurement[0]) + 1)
			if settings.has_key("plot"):
				fit_params = measurements.potential_params(measurement)
				fit_line = measurements.pair_potential(fit_params, r)
				error_linestyle = 'o' + i["linestyle"]["colour"]
				plot_linestyle = i["linestyle"]["colour"] \
				  + i["linestyle"]["style"]
				pl.errorbar(r, measurement[0], yerr = measurement[1],
							fmt = error_linestyle, label = i["label"])
				pl.plot(r_fit, fit_line, plot_linestyle,
						label = i["label"] + " fit")

			if settings.has_key("store"):
				output.append(r)
				output.append(measurements[0])
				output.append(measurements[1])

	if settings.has_key("plot"):
		pl.show()

	if settings.has_key("store"):
		save(settings["store"]["filename"], output)
	
