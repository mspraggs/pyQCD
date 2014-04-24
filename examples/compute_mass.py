"""
Here we analyze the correlator data generated by the compute_correlators.py
script, plotting the effective mass curve and the fitted mass.
"""

import numpy as np
import matplotlib.pyplot as plt

import pyQCD

if __name__ == "__main__":
    
    # First load the data from the zip archive
    data = pyQCD.DataSet.load("4c8_correlators.zip")
    
    # Then we'll need an estimate of the error in the correlators to supply
    # to the fitting function
    data_mean, data_err = data.statistics()
    
    # Extract the correlator error from the data_std TwoPoint object. We're
    # interested in the pseudoscalar meson here, so we need to use the label
    # "g5_g5" to refer to the appropriate correlator. If correlators are
    # generated from pyQCD.Propagator objects using
    # compute_all_meson_correlators, then the labels used in the TwoPoint object
    # correspond to the gamma matrices that form the spin structures used as
    # interpolators. For example, the interpolators for the pion are both
    # \gamma_5 matrices, so the corresponding label will be "g5_g5". Also note
    # that since the data generated by compute_correlators.py contains multiple
    # smearing combinations, the type of source and sink must also be specified.
    corr_err = data_err.get_correlator("g5_g5", source_type="point_point",
                                       sink_type="point_point")
    
    # Here's the fit range we'll use, as we'll need it more than once
    fit_range = [2, 6]
    
    # Now we'll fit the lowest lying state of the pseudoscalar correlator. We do
    # the fit under the jackknife to get an estimation of the error. The
    # jackknife function takes the measurement function as its first argument,
    # and the arguments to the measurement function are specified using the args
    # keyword argument. Here we specify a fit range from the 2nd to 6th
    # timeslices, and we use the same correlator specifiers as when we extracted
    # the error in the correlator, though this time we need to specify the
    # masses and momentum of the constituent quarks and meson, because we cannot
    # use keyword args to skip optional parameters.
    mass_mean, mass_err = data.jackknife(pyQCD.TwoPoint.compute_energy,
                                         args=[fit_range, [1., 1.],
                                               corr_err, "g5_g5", (0.4, 0.03),
                                               (0, 0, 0), "point_point",
                                               "point_point"])
    
    # To plot the effective mass, we'll need to compute it under the jackknife
    # again
    effmass_mean, effmass_err = data.jackknife(pyQCD.TwoPoint.compute_effmass,
                                               args=["g5_g5", (0.4, 0.03),
                                                     (0, 0, 0), "point_point",
                                                     "point_point"])
    
    t = np.arange(effmass_mean.size)    
    plt.errorbar(t, effmass_mean, effmass_err, fmt='o', capsize=0)
    
    # Plot the mass over the fitting range, along with error margins
    fit_range[1] -= 1
    plt.plot(fit_range, [mass_mean, mass_mean], 'b--')
    plt.fill_between(fit_range, [mass_mean - mass_err, mass_mean - mass_err],
                     [mass_mean + mass_err, mass_mean + mass_err], alpha=0.5)
    
    plt.xlabel("$t$", fontsize=18)
    plt.ylabel("$m_\mathrm{err}(t)$", fontsize=18)
    
    plt.show()