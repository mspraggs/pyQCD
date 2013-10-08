import pylab as pl
from numpy import save, load
import statistics
import measurements

def auto_correlation(settings):
	"""Load the input file(s) in the autocorrelation settings, calculate the
	autocorrelation function and plot and store it if necessary.

	The settings dictionary should be a nested dictionary with the same
	structure as the xml settings."""

	inputs = settings["input"]

	output = []

	for i in inputs:
		if i["type"] != 0:
			print("Error, invalid input data. Skipping.")
		else:
			input_data = load(i["filename"])
			measurement = measurements.auto_correlation(input_data)

			t = pl.arange(pl.size(measurement, axis = 0))
			if settings.has_key("plot"):
				linestyle = i["linestyle"]["colour"] + i["linestyle"]["style"]
				pl.plot(t, measurement, linestyle, label = i["label"])

			if settings.has_key("store"):
				output.append(t)
				output.append(measurement)

	if settings.has_key("plot"):
		pl.xlabel(settings["plot"]["xlabel"])
		pl.ylabel(settings["plot"]["ylabel"])
		pl.title(settings["plot"]["title"])
		pl.legend(loc=0)
		pl.show()

		if settings["plot"].has_key("filename"):
			pl.savefig(settings["plot"]["filename"])

	if settings.has_key("store"):
		save(settings["store"]["filename"], output)

def pair_potential(settings):
	"""Load the input file(s) in the pair potential settings, calculate the
	pair potential function and plot and store it if necessary.

	The settings dictionary should be a nested dictionary with the same
	structure as the xml settings."""

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
				fit_params = measurements.potential_params(measurement[0])

				r_fit = pl.arange(0.1, pl.size(measurement[0]) + 1, 0.1)
				fit_line = measurements.pair_potential(fit_params, r_fit)
				
				error_linestyle = 'o' + i["linestyle"]["colour"]
				plot_linestyle = i["linestyle"]["colour"] \
				  + i["linestyle"]["style"]
				pl.errorbar(r, measurement[0], yerr = measurement[1],
							fmt = error_linestyle, label = i["label"])
				pl.plot(r_fit, fit_line, plot_linestyle,
						label = i["label"] + " fit")

			if settings.has_key("store"):
				output.append(r)
				output.append(measurement[0])
				output.append(measurement[1])

	if settings.has_key("plot"):
		pl.xlabel(settings["plot"]["xlabel"])
		pl.ylabel(settings["plot"]["ylabel"])
		pl.title(settings["plot"]["title"])
		pl.legend(loc=0)
		pl.show()

		if settings["plot"].has_key("filename"):
			pl.savefig(settings["plot"]["filename"])

	if settings.has_key("store"):
		save(settings["store"]["filename"], output)


def lattice_spacing(settings):
	"""Load the input file(s) in the lattice spacing settings, calculate the
	lattice spacing function and plot and store it if necessary.

	The settings dictionary should be a nested dictionary with the same
	structure as the xml settings."""

	inputs = settings["input"]

	output = []

	for i in inputs:
		if i["type"] != 1:
			print("Error, invalid input data. Skipping.")
		else:
			input_data = load(i["filename"])
			measurement = statistics \
			  .bootstrap_measurement(input_data,
									 measurements.calculate_spacing,
									 settings["num_bootstraps"],
									 settings["bin_size"])

			print("Lattice spacing: %f +/- %f fm" % measurement)

			if settings.has_key("store"):
				output.append(measurement[0])
				output.append(measurement[1])

	if settings.has_key("store"):
		save(settings["store"]["filename"], output)
