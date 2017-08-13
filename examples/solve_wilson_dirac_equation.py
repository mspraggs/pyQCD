from pyQCD.algorithms import conjugate_gradient_eoprec
from pyQCD.core import EvenOddLayout, LatticeColourVector
from pyQCD.fermions import WilsonFermionAction
from pyQCD.gauge import cold_start

# Gauge field must be even-odd partitioned for even-odd solver to work
gauge_field = cold_start([16, 8, 8, 8])
lexico_layout = gauge_field.layout
even_odd_layout = EvenOddLayout(lexico_layout.shape)
gauge_field.change_layout(even_odd_layout)

# Create the Wilson action with dimensionless mass of 0.4 and periodic
# boundary conditions
fermion_action = WilsonFermionAction(0.4, gauge_field, [0] * 4)

# Create a point source
rhs = LatticeColourVector(lexico_layout, 4)
rhs[0, 0, 0, 0, 0, 0] = 1.0
rhs.change_layout(even_odd_layout)

# Solve Dirac equation
sol, num_iterations, err = conjugate_gradient_eoprec(
    fermion_action, rhs, 1000, 1e-10)
print("Finished solving after {} iterations. Solution has error = {}"
      .format(num_iterations, err))

# Change layout of sol back to lexicographic to enable numpy-functionality
sol.change_layout(lexico_layout)
# Print solution at (t, x, y, z) = (0, 0, 0, 0) and alpha = 0
print(sol[0, 0, 0, 0, 0])
