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

gamma4 = gamma0

gamma5 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, -1]],
                   dtype = np.complex)

gammas = [gamma0, gamma1, gamma2, gamma3, gamma4, gamma5]

Gammas = {0: id4,
          '1': id4,
          'a0': id4,
          1: gamma1,
          'g1': gamma1,
          'rho_x': gamma1,
          2: gamma2,
          'g2': gamma2,
          'rho_y': gamma2,
          3: np.dot(gamma1, gamma2),
          'g1g2': np.dot(gamma1, gamma2),
          'b1_z': np.dot(gamma1, gamma2),
          4: gamma3,
          'g3': gamma3,
          'rho_z': gamma3,
          5: np.dot(gamma1, gamma3),
          'g1g3': np.dot(gamma1, gamma3),
          'b1_y': np.dot(gamma1, gamma3),
          6: np.dot(gamma2, gamma3),
          'g2g3': np.dot(gamma2, gamma3),
          'b1_x': np.dot(gamma2, gamma3),
          7: np.dot(gamma5, gamma4),
          'g5g4': np.dot(gamma5, gamma4),
          'pion_2': np.dot(gamma5, gamma4),
          8: gamma4,
          'g4': gamma4,
          'a0_2': gamma4,
          9: np.dot(gamma1, gamma4),
          'g1g4': np.dot(gamma1, gamma4),
          'rho_x_2': np.dot(gamma1, gamma4),
          10: np.dot(gamma2, gamma4),
          'g2g4': np.dot(gamma2, gamma4),
          'rho_y_2': np.dot(gamma2, gamma4),
          11: np.dot(gamma3, gamma5),
          'g3g5': np.dot(gamma3, gamma5),
          'a1_z': np.dot(gamma3, gamma5),
          12: np.dot(gamma3, gamma4),
          'g3g4': np.dot(gamma3, gamma4),
          'rho_z_2': np.dot(gamma3, gamma4),
          13: -np.dot(gamma2, gamma5),
          '-g2g5': -np.dot(gamma2, gamma5),
          'a1_y': -np.dot(gamma2, gamma5),
          14: np.dot(gamma1, gamma5),
          'g1g5': np.dot(gamma1, gamma5),
          'a1_x': np.dot(gamma1, gamma5),
          15: gamma5,
          'g5': gamma5,
          'pion': gamma5
          }

mesons = ['a0', 'rho_x', 'rho_y', 'b1_z', 'rho_z', 'b1_y', 'b1_x',
          'pion_2', 'a0_2', 'rho_x_2', 'rho_y_2', 'a1_z', 'rho_z_2',
          'a1_y', 'a1_x', 'pion']
    
baryons_degenerate = ['proton', 'lambda', 'delta', 'proton', 'lambda', 'delta',
                      'proton', 'lambda', 'delta', 'proton', 'proton', 'proton',
                      'lambda', 'xi', 'lambda', 'xi', 'proton_negpar']
    
baryons_m1m2 = ['sigma', 'lambda', 'sigma_st', 'sigma', 'lambda', 'sigma_st',
                'sigma', 'lambda', 'sigma_st', 'sigma', 'sigma', 'sigma',
                'lambda', 'xi', 'lambda', 'xi', 'sigma_negpar']
    
baryons_m2m1 = ['xi', 'lambda', 'xi_st', 'xi', 'lambda', 'xi_st',
                'xi', 'lambda', 'xi_st', 'xi', 'xi', 'xi',
                'lambda', 'sigma', 'lambda', 'sigma', 'xi_negpar']
    
interpolators = ['1', 'g1', 'g2', 'g1g2', 'g3', 'g1g3', 'g2g3', 'g5g4',
                 'g4', 'g1g4', 'g2g4', 'g3g5', 'g3g4', '-g2g5', 'g1g5',
                 'g5']
