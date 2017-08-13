from pyQCD.algorithms import Heatbath
from pyQCD.gauge import WilsonGaugeAction, average_plaquette, cold_start
 
# Create a gauge field with all matrices set to the identity, laid out
# lexicographically
gauge_field = cold_start([16, 8, 8, 8])
# Create a Wilson gauge action instance with beta = 5.5
action = WilsonGaugeAction(5.5, gauge_field.layout)
# Create a heatbath update instance
heatbath_updater = Heatbath(gauge_field.layout, action)
 
# Update the gauge field 100 times
heatbath_updater.update(gauge_field, 100)
# Show link at (t, x, y, z) = (0, 0, 0, 0) and mu = 0
print("Average plaquette = {}".format(average_plaquette(gauge_field)))
