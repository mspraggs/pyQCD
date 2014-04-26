"""
Here we use scipy to compute the eigenvalue spectrum of the domain wall
Dirac operator
"""

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

import pyQCD

if __name__ == "__main__":
    
    # Create a lattice object, 4^3 x 8 in size. No updates, so this is
    # a tree-level calculation.
    lattice = pyQCD.Lattice()

    # Now we construct a scipy sparse linear operator.
    Ls = 4
    # First here's the matrix-vector function.
    def matvec(psi):
        # Here the quark mass is set to 0.4
        return lattice.apply_dwf_dirac(psi, 0.4, 1.8, Ls).flatten()

    # We'll also need the size of the vectors on which matvec operates
    N = 12 * np.prod(lattice.shape) * Ls
        
    # Now create the linear operator
    linop = spla.LinearOperator((N, N), matvec, dtype=np.complex)
    
    # Compute the eigenvalues
    eigvals, eigvecs = spla.eigs(linop, k=10)
    
    # Plot the eigenvalues.
    plt.plot(eigvals.real, eigvals.imag, 'o')
    plt.show()
