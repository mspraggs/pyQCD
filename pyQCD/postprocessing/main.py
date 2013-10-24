import pylab as pl
from numpy import save, load
import statistics
import measurements

def auto_correlation(settings):
    """Load the input file(s) in the autocorrelation settings, calculate the
    autocorrelation function and store it.

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
            output.append(t)
            output.append(measurement)

    save(settings["filename"], output)

def pair_potential(settings):
    """Load the input file(s) in the pair potential settings, calculate the
    pair potential function and store it.

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
            output.append(r)
            output.append(measurement[0])
            output.append(measurement[1])
            
    output = pl.array(output).T
    
    print("r\tV(r)\tdV(r)")
    
    for point in output:
        print("{}\t{}\t{}".format(point[0],point[1],point[2]))

    save(settings["filename"], output)


def lattice_spacing(settings):
    """Load the input file(s) in the lattice spacing settings, calculate the
    lattice spacing function and store it.

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
            output.append(measurement[0])
            output.append(measurement[1])
            
    save(settings["filename"], output)

def correlators(settings):
    """Open the input correlator files in the correlator settings and
    compute the correlators"""

    pass
