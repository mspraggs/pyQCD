import numpy as np
import itertools

id4 = np.identity(4)

gamma0 = np.array([[0, 0, 1, 0],
				   [0, 0, 0, 1],
				   [1, 0, 0, 0],
				   [0, 1, 0, 0]],
				   dtype = np.complex)

gamma1 = np.array([[0, 0, 0, -1j],
				   [0, 0, -1j, 0],
				   [0, 1j, 0, 0],
				   [1j, 0, 0, 0]],
				   dtype = np.complex)

gamma2 = np.array([[0, 0, 0, -1],
				   [0, 0, 1, 0],
				   [0, 1, 0, 0],
				   [-1, 0, 0, 0]],
				   dtype = np.complex)

gamma3 = np.array([[0, 0, -1j, 0],
				   [0, 0, 0, 1j],
				   [1j, 0, 0, 0],
				   [0, -1j, 0, 0]],
				   dtype = np.complex)

gamma4 = gamma1

gamma5 = np.array([[1, 0, 0, 0],
				   [0, 1, 0, 0],
				   [0, 0, -1, 0],
				   [0, 0, 0, -1]],
				   dtype = np.complex)

gammas = [gamma0, gamma1, gamma2, gamma3, gamma4, gamma5]

Gammas = {
	'scalar': [id4,
			   gamma4],
	'pseudoscalar': [gamma5,
					 np.dot(gamma4, gamma5)],
	'vector': [gamma1,
			   gamma2,
			   gamma3,
			   np.dot(gamma4, gamma1),
			   np.dot(gamma4, gamma2),
			   np.dot(gamma4, gamma3)],
	'axial_vector': [np.dot(gamma1, gamma5),
					 np.dot(gamma2, gamma5),
					 np.dot(gamma3, gamma5)],
	'tensor': [np.dot(gamma1, gamma2),
			   np.dot(gamma1, gamma3),
			   np.dot(gamma2, gamma3)]}
