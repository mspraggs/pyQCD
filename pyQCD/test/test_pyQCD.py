import os
import shutil
import itertools
import string
import zipfile

import pytest
import numpy as np
import numpy.random as npr
import scipy.linalg as spla

from pyQCD import *

try:
    test_lattice = Lattice()
    lattice_exists = True
except NameError:
    lattice_exists = False
        
data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

def create_fullpath(fname):
    
    return "{}/{}".format(data_dir, fname)
        
def random_complex(shape):
    
    return npr.random(shape) * np.exp(2 * np.pi * 1j * npr.random(shape))

def random_variable_name():
    
    allowed_chars = string.uppercase + string.lowercase + string.digits
    
    name = (string.uppercase + string.lowercase)[npr.randint(52)]
    
    character_indices = [npr.randint(62) for i in xrange(npr.randint(30))]
    
    for i, index in enumerate(character_indices):
        if npr.random() < 0.2:
            name += "_"
        name += allowed_chars[index]
        
    return name

def make_random_su(N):
    
    A = spla.expm(2 * np.pi * 1j * npr.random((N, N)))
    B = (A - np.trace(A) * np.identity(N) / N)
    C = 0.5 * (B - np.conj(B.T))
    
    return spla.expm(C)

def test_random_su():
    
    # NB if this fails, many of the tests below will probably also fail
    for i in xrange(2, 5):
        U = make_random_su(i)
        
        determinant = spla.det(U)
        assert np.abs(determinant - 1.0) < 5e-15
        
        UUdagger = np.dot(U, np.conj(U.T))
        assert (np.abs(UUdagger - np.identity(i)) 
                < 1e-12 * np.ones((i, i))).all()
        
def make_links(T, L):
        
    links = [[t, x, y, z, mu]
             for t in xrange(T)
             for x in xrange(L)
             for y in xrange(L)
             for z in xrange(L)
             for mu in xrange(4)]
    
    return links

def test_make_links():
    # If this test fails, expect the other tests below to fail
    links = make_links(8, 4)
    
    assert len(links) == 2048
    
    for link in links:
        assert links.count(link) == 1
        
def make_sites(T, L):
        
    sites = [[t, x, y, z]
             for t in xrange(T)
             for x in xrange(L)
             for y in xrange(L)
             for z in xrange(L)]
    
    return sites

def test_make_sites():
    # If this test fails, expect the other tests below to fail
    sites = make_sites(8, 4)
    
    assert len(sites) == 512
    
    for link in sites:
        assert sites.count(link) == 1
        
def random_su3_transform(lattice):
    
    links = make_links(lattice.T, lattice.L)
    
    U = make_random_su(3)
    
    for link in links:
        matrix = lattice.get_link(link)
        new_matrix = np.dot(U, np.dot(matrix, np.conj(U.T)))
        lattice.set_link(link, new_matrix)

if lattice_exists:
    def test_random_su3_transform():
        
        lattice = Lattice()
        
        random_su3_transform(lattice)
        
        links = make_links(lattice.T, lattice.L)
        
        for link in links:
            determinant = spla.det(lattice.get_link(link))
            assert np.allclose(determinant, 1.0)
        
            UUdagger = np.dot(lattice.get_link(link),
                              np.conj(lattice.get_link(link).T))
            assert np.allclose(UUdagger, np.identity(3))
        
class floatWrapper(float):
    
    def save(self, filename):
        
        if filename[-4:] != ".npz":
            filename += ".npz"
            
        with open(filename, "w") as f:
            f.write(self.__repr__())
    
    @classmethod
    def load(self, filename):
        
        if filename[-4:] != ".npz":
            filename += ".npz"        
        
        with open(filename) as f:
            value = floatWrapper(f.read())
        
        return value
    
    def __add__(self, x):
        return floatWrapper(float.__add__(self, x))
    
    def __sub__(self, x):
        return floatWrapper(float.__sub__(self, x))
    
    def __mul__(self, x):
        return floatWrapper(float.__mul__(self, x))
    
    def __div__(self, x):
        return floatWrapper(float.__div__(self, x))
    
    def __pow__(self, x):
        return floatWrapper(float.__pow__(self, x))
    
if lattice_exists:
    class TestLattice:
    
        def test_block_calculation(self):
            with pytest.raises(ValueError):
                lattice = Lattice(L=3, T=5)
            
        def test_get_link(self):
            lattice = Lattice()
            
            links = make_links(lattice.T, lattice.L)
            
            offsets = [[t, x, y, z, 0]
                   for t in [0, lattice.T]
                   for x in [0, lattice.L]
                   for y in [0, lattice.L]
                   for z in [0, lattice.L]]
        
            # Check that periodic boundary conditions work
            # Note if this test fails, most of the others will too because they
            # depend on get_link (or the C++ version)
        
            for link in links:
                for offset in offsets:
                    link_offset = map(lambda x, y: x + y, link, offset)
                    
                    assert (lattice.get_link(link) 
                            == lattice.get_link(link_offset)).all()

        def test_set_link(self):
            
            lattice = Lattice()
            links = make_links(lattice.T, lattice.L)
            
            # Check that each link can be set properly
            
            for link in links:
                U = make_random_su(3)
                lattice.set_link(link, U)
                assert (lattice.get_link(link) == U).all()
            
        def test_get_config(self):
            
            lattice = Lattice()
            lattice.update()

            config = lattice.get_config()
            
            assert config.L == lattice.L
            assert config.T == lattice.T
            assert config.beta == lattice.beta
            assert config.ut == lattice.ut
            assert config.us == lattice.us
            assert config.chi == lattice.chi
            assert config.action == lattice.action
            
            expected_shape = (lattice.T, lattice.L, lattice.L, lattice.L, 4, 3, 3)
            
            assert config.data.shape == expected_shape
            
            links = make_links(lattice.T, lattice.L)
            
            for link in links:
                assert (lattice.get_link(link) == config.data[tuple(link)]).all()

        def test_set_config(self):
            # This could fail if the Config constructor doesn't work
            lattice = Lattice()
            
            # Generate some fake su3 config
            config_shape = (lattice.T, lattice.L, lattice.L, lattice.L, 4, 3, 3)
            data = np.zeros(config_shape, dtype=np.complex)

            links = make_links(lattice.T, lattice.L)
            
            for t, x, y, z, mu in links:
                data[t, x, y, z] = make_random_su(3)

            config = Config(data, lattice.L, lattice.T, lattice.beta, lattice.ut,
                            lattice.us, lattice.chi, lattice.action)
            
            lattice.set_config(config)
            
            assert lattice.L == config.L
            assert lattice.T == config.T
            assert lattice.beta == config.beta
            assert lattice.ut == config.ut
            assert lattice.us == config.us
            assert lattice.chi == config.chi
            assert lattice.action == config.action
            
            for link in links:
                assert (lattice.get_link(link) == config.data[tuple(link)]).all()
                
        def test_save_config(self):
        
            lattice = Lattice()
            
            lattice.save_config("test_config.npz")
            
            assert os.path.exists("test_config.npz")
            
            test_config = np.load("test_config.npz")
            
            assert "data" in test_config.files
            assert "header" in test_config.files

        def test_load_config(self):
            
            lattice = Lattice()

            lattice.load_config("test_config.npz")
            
            os.unlink("test_config.npz")
            
        def test_update(self):
    
            # Generate some configs and save the raw data
            for gauge_action in dicts.gauge_actions.keys():
                for update_method in dicts.update_methods.keys():
                    for rand_seed in [0, 1, 2]:
                
                        filename = "config_{}_{}_{}" \
                          .format(gauge_action,
                                  update_method,
                                  rand_seed)
                
                        lattice = Lattice(rand_seed=rand_seed,
                                          action=gauge_action,
                                          update_method=update_method)
                        lattice.update()
                        
                        config_data = np.load(create_fullpath(filename))
                        
                        assert np.allclose(lattice.get_config().data, config_data)

        def test_next_config(self):
            
            lattice = Lattice(rand_seed=0, n_cor=100)
            
            lattice.next_config()
            links = make_links(lattice.T, lattice.L)
            
            for link in links:
                matrix = lattice.get_link([0, 0, 0, 0, 0])
                # Test that the gauge links are SU3
                determinant = spla.det(matrix)
                assert np.abs(spla.det(matrix) - 1.0) < 1e-12
            
                UUdagger = np.dot(matrix, np.conj(matrix.T))
                assert (np.abs(UUdagger - np.identity(3)) 
                        < 1e-12 * np.ones((3, 3))).all()
            
        def test_thermalize(self):
            
            lattice = Lattice(T=8, L=8)
            
            lattice.thermalize(200)
            
            assert np.allclose(lattice.get_av_plaquette(), 0.5, atol=0.1)

        def test_get_plaquette(self):
            
            lattice = Lattice()
            
            sites = make_sites(lattice.T, lattice.L)
            
            for site in sites:
                for mu, nu in zip([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]):
                    assert lattice.get_plaquette(site, mu, nu) == 1.0
                        
            for i in xrange(100):
                lattice.update()
            
            # Test for gauge invariance  
            plaquettes = [lattice.get_plaquette(site, mu, nu)
                          for mu in [0, 0, 0, 1, 1, 2]
                          for nu in [1, 2, 3, 2, 3, 3]
                          for site in sites]

            random_su3_transform(lattice)
            
            transformed_plaquettes = [lattice.get_plaquette(site, mu, nu)
                                      for mu in [0, 0, 0, 1, 1, 2]
                                      for nu in [1, 2, 3, 2, 3, 3]
                                      for site in sites]

            for P1, P2 in zip(plaquettes, transformed_plaquettes):
                assert np.allclose(P1, P2)

        def test_get_rectangle(self):
                
            lattice = Lattice()
            
            sites = make_sites(lattice.T, lattice.L)
            
            for site in sites:
                for mu, nu in zip([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]):
                    assert lattice.get_rectangle(site, mu, nu) == 1.0
                
            for i in xrange(100):
                lattice.update()
            
            # Test for gauge invariance  
            rectangles = [lattice.get_rectangle(site, mu, nu)
                          for mu in [0, 0, 0, 1, 1, 2]
                          for nu in [1, 2, 3, 2, 3, 3]
                          for site in sites]

            random_su3_transform(lattice)
            
            transformed_rectangles = [lattice.get_rectangle(site, mu, nu)
                                      for mu in [0, 0, 0, 1, 1, 2]
                                      for nu in [1, 2, 3, 2, 3, 3]
                                      for site in sites]

            for R1, R2 in zip(rectangles, transformed_rectangles):
                assert np.allclose(R1, R2)

        def test_get_twist_rect(self):
        
            lattice = Lattice()
        
            sites = make_sites(lattice.T, lattice.L)
        
            for site in sites:
                for mu, nu in zip([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]):
                    assert lattice.get_rectangle(site, mu, nu) == 1.0
                
            for i in xrange(100):
                lattice.update()
        
            # Test for gauge invariance  
            twist_rectangles = [lattice.get_twist_rect(site, mu, nu)
                                for mu in [0, 0, 0, 1, 1, 2]
                                for nu in [1, 2, 3, 2, 3, 3]
                                for site in sites]

            random_su3_transform(lattice)
        
            transformed_twist_rectangles = [lattice.get_twist_rect(site, mu, nu)
                                            for mu in [0, 0, 0, 1, 1, 2]
                                            for nu in [1, 2, 3, 2, 3, 3]
                                            for site in sites]

            for T1, T2 in zip(twist_rectangles, transformed_twist_rectangles):
                assert np.allclose(T1, T2)

        def test_get_wilson_loop(self):
        
            lattice = Lattice()
            
            sites = make_sites(lattice.T, lattice.L)
            
            T_range = [1, lattice.T / 2, lattice.T - 1]
            L_range = [1, lattice.L / 2, lattice.L - 1]
            
            for site in sites:
                for r, t in itertools.product(L_range, T_range):
                    for dim in range(1, 4):
                        for n in range(3):
                            assert lattice.get_wilson_loop(site, r, t,
                                                           dim, n, 0.5) == 1.0
                
            for i in xrange(100):
                lattice.update()
            
            # Test for gauge invariance - only valid for non-smeared gauge fields
            wilson_loops \
              = [lattice.get_wilson_loop(site, r, t, dim)
                 for dim in range(1, 4)
                 for r, t in itertools.product(L_range, T_range)
                 for site in sites]

            random_su3_transform(lattice)
            
            transformed_wilson_loops \
              = [lattice.get_wilson_loop(site, r, t, dim)
                 for dim in range(1, 4)
                 for r, t in itertools.product(L_range, T_range)
                 for site in sites]

            for W1, W2 in zip(wilson_loops, transformed_wilson_loops):
                assert np.allclose(W1, W2)
            
            # Test to make sure no smearing is applied.
            W1 = lattice.get_wilson_loop([0, 0, 0, 0], 4, 4, 1)
            for n in xrange(10):
                W2 = lattice.get_wilson_loop([0, 0, 0, 0, 0], 4, 4, 1, 0,
                                             0.1 * n)
                assert W1 == W2

        def test_get_av_plaquette(self):
            
            lattice = Lattice(rand_seed=0, update_method="heatbath")
            lattice.update()
            
            assert np.allclose(lattice.get_av_plaquette(), 0.6744055385048071)
            
            random_su3_transform(lattice)
            
            assert np.allclose(lattice.get_av_plaquette(), 0.6744055385048071)

        def test_get_av_rectangle(self):
            
            lattice = Lattice(rand_seed=0, update_method="heatbath")
            lattice.update()
            
            assert np.allclose(lattice.get_av_rectangle(), 0.5093032901600738)
            
            random_su3_transform(lattice)
            
            assert np.allclose(lattice.get_av_rectangle(), 0.5093032901600738)

        def test_get_av_wilson_loop(self):
            
            lattice = Lattice(rand_seed=0)
            lattice.update()
            
            W = lattice.get_av_wilson_loop(4, 4)
            assert np.allclose(W, 0.2883925516552541)

            random_su3_transform(lattice)
            
            W = lattice.get_av_wilson_loop(4, 4)
            assert np.allclose(W, 0.2883925516552541)
            
        def test_get_wilson_loops(self):
            
            lattice = Lattice()
            
            for n in xrange(3):
                wilson_loops = lattice.get_wilson_loops(n, 0.5)
            
                for r in xrange(lattice.L):
                    for t in xrange(lattice.T):
                        assert wilson_loops.data[r, t] \
                          == lattice.get_av_wilson_loop(r, t, n, 0.5)

        def test_get_wilson_propagator(self):
            
            lattice = Lattice()
            config_data \
              = np.load(create_fullpath("config_wilson_heatbath_0.npy"))
            config = Config(config_data, 4, 8, 5.5, 1.0, 1.0, 1.0,
                            "wilson")
            lattice.set_config(config)
            
            smearing_combinations \
              = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
              
            func = Lattice.get_wilson_propagator
                
            for solver_method in dicts.solver_methods.keys():
                for smearing_type in dicts.smearing_types.keys():
                    for n_link_s, n_source_s, n_sink_s in smearing_combinations:
                        
                        prop = func(lattice, 0.4,
                                    num_field_smears=n_link_s,
                                    field_smearing_param=0.4,
                                    source_smear_type=smearing_type,
                                    num_source_smears=n_source_s,
                                    source_smearing_param=0.4,
                                    sink_smear_type=smearing_type,
                                    num_sink_smears=n_sink_s,
                                    sink_smearing_param=0.4,
                                    solver_method=solver_method)
                    
                        filename = "prop_wilson_{}_{}_{}_{}_{}.npy" \
                          .format(solver_method, smearing_type, n_link_s,
                                  n_source_s, n_sink_s)
                        
                        actual_prop = np.load(create_fullpath(filename))
                        
                        assert np.allclose(actual_prop, prop.data)

        def test_get_hamberwu_propagator(self):
            
            lattice = Lattice()
            config_data \
              = np.load(create_fullpath("config_wilson_heatbath_0.npy"))
            config = Config(config_data, 4, 8, 5.5, 1.0, 1.0, 1.0,
                                  "wilson")
            lattice.set_config(config)
            
            smearing_combinations \
              = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
              
            func = Lattice.get_hamberwu_propagator
                
            for solver_method in dicts.solver_methods.keys():
                for smearing_type in dicts.smearing_types.keys():
                    for n_link_s, n_source_s, n_sink_s in smearing_combinations:
                        
                        prop = func(lattice, 0.4,
                                    num_field_smears=n_link_s,
                                    field_smearing_param=0.4,
                                    source_smear_type=smearing_type,
                                    num_source_smears=n_source_s,
                                    source_smearing_param=0.4,
                                    sink_smear_type=smearing_type,
                                    num_sink_smears=n_sink_s,
                                    sink_smearing_param=0.4,
                                    solver_method=solver_method)
                    
                        filename = "prop_hamber-wu_{}_{}_{}_{}_{}.npy" \
                          .format(solver_method, smearing_type, n_link_s,
                                  n_source_s, n_sink_s)
                        
                        actual_prop = np.load(create_fullpath(filename))
                        
                        assert np.allclose(actual_prop, prop.data)
                        
        def test_invert_wilson_dirac(self):
            
            lattice = Lattice()
            config_data \
              = np.load(create_fullpath("config_wilson_heatbath_0.npy"))
            config = Config(config_data, 4, 8, 5.5, 1.0, 1.0, 1.0,
                                  "wilson")
            lattice.set_config(config)
            
            psi = np.zeros([8, 4, 4, 4, 4, 3])
            psi[0, 0, 0, 0, 0, 0] = 1.0
            
            for solver_method in dicts.solver_methods.keys():
                
                eta = lattice.invert_wilson_dirac(psi, 0.4,
                                                  solver_method=solver_method)
            
                filename = "spinor_wilson_{}.npy" \
                  .format(solver_method)
                
                expected_eta = np.load(create_fullpath(filename))
                
                assert np.allclose(eta, expected_eta)
                        
        def test_invert_hamberwu_dirac(self):
            
            lattice = Lattice()
            config_data \
              = np.load(create_fullpath("config_wilson_heatbath_0.npy"))
            config = Config(config_data, 4, 8, 5.5, 1.0, 1.0, 1.0,
                                  "wilson")
            lattice.set_config(config)
            
            psi = np.zeros([8, 4, 4, 4, 4, 3])
            psi[0, 0, 0, 0, 0, 0] = 1.0
            
            for solver_method in dicts.solver_methods.keys():
                
                eta = lattice.invert_hamberwu_dirac(psi, 0.4,
                                                    solver_method=solver_method)
            
                filename = "spinor_hamber-wu_{}.npy" \
                  .format(solver_method)
                
                expected_eta = np.load(create_fullpath(filename))
                
                assert np.allclose(eta, expected_eta)
                        
        def test_invert_dwf_dirac(self):
            
            lattice = Lattice()
            config_data \
              = np.load(create_fullpath("config_wilson_heatbath_0.npy"))
            config = Config(config_data, 4, 8, 5.5, 1.0, 1.0, 1.0,
                                  "wilson")
            lattice.set_config(config)
            
            psi = np.zeros([4, 8, 4, 4, 4, 4, 3])
            psi[0, 0, 0, 0, 0, 0, 0] = 1.0
            
            for fermion_action in dicts.fermion_actions.keys():
                
                eta = lattice.invert_dwf_dirac(psi, 0.4, 1.6, 4,
                                               kernel = fermion_action)
            
                filename = "spinor_dwf_{}_conjugate_gradient.npy" \
                  .format(fermion_action)
                
                expected_eta = np.load(create_fullpath(filename))
                
                assert np.allclose(eta, expected_eta)
            
        def test_apply_wilson_dirac(self):
            
            lattice = Lattice()
            config_data \
              = np.load(create_fullpath("config_wilson_heatbath_0.npy"))
            config = Config(config_data, 4, 8, 5.5, 1.0, 1.0, 1.0,
                            "wilson")
            lattice.set_config(config)
            
            psi = np.zeros([8, 4, 4, 4, 4, 3])
            psi[0, 0, 0, 0, 0, 0] = 1.0
            
            eta = lattice.apply_wilson_dirac(psi, 0.4)
            expected_eta = np.load(create_fullpath("Dpsi_wilson.npy"))
            
            assert np.allclose(eta, expected_eta)
            
        def test_apply_hamberwu_dirac(self):
            
            lattice = Lattice()
            config_data \
              = np.load(create_fullpath("config_wilson_heatbath_0.npy"))
            config = Config(config_data, 4, 8, 5.5, 1.0, 1.0, 1.0,
                            "wilson")
            lattice.set_config(config)
            
            psi = np.zeros([8, 4, 4, 4, 4, 3])
            psi[0, 0, 0, 0, 0, 0] = 1.0
            
            eta = lattice.apply_hamberwu_dirac(psi, 0.4)
            expected_eta = np.load(create_fullpath("Dpsi_hamber-wu.npy"))
            
            assert np.allclose(eta, expected_eta)
            
        def test_apply_dwf_dirac(self):
            
            lattice = Lattice()
            config_data \
              = np.load(create_fullpath("config_wilson_heatbath_0.npy"))
            config = Config(config_data, 4, 8, 5.5, 1.0, 1.0, 1.0,
                            "wilson")
            lattice.set_config(config)
            
            psi = np.zeros([4, 8, 4, 4, 4, 4, 3])
            psi[0, 0, 0, 0, 0, 0, 0] = 1.0
            
            for fermion_action in dicts.fermion_actions.keys():
                eta = lattice.apply_dwf_dirac(psi, 0.4, 1.6, 4, fermion_action)
                expected_eta \
                  = np.load(create_fullpath("Dpsi_dwf_{}.npy".format(fermion_action)))
            
                assert np.allclose(eta, expected_eta)
            
        def test_apply_jacobi_smearing(self):
            
            lattice = Lattice()
            config_data \
              = np.load(create_fullpath("config_wilson_heatbath_0.npy"))
            config = Config(config_data, 4, 8, 5.5, 1.0, 1.0, 1.0,
                            "wilson")
            lattice.set_config(config)
            
            psi = np.zeros([8, 4, 4, 4, 4, 3])
            psi[0, 0, 0, 0, 0, 0] = 1.0
            
            eta = lattice.apply_jacobi_smearing(psi, 1, 0.5)
            expected_eta = np.load(create_fullpath("smeared_source_jacobi.npy"))
            
            assert np.allclose(eta, expected_eta)
    
        def test_get_av_link(self):
            
            lattice = Lattice()

            assert np.allclose(lattice.get_av_link(), 1.0)

class TestObservable:
    
    def test_init(self):
        ob = Observable()
        ob = Observable(random_complex(100))

    def test_save(self):
        
        data = random_complex(100)
        
        ob = Observable(data)
        ob.save("test_observable.npz")
        
        assert os.path.exists("test_observable.npz")

        test_ob = np.load("test_observable.npz")
        
        assert "data" in test_ob.files
        assert "header" in test_ob.files
        
        assert (test_ob["data"] == data).all()
        assert test_ob["header"] == {}

    def test_load(self):
        
        ob = Observable.load("test_observable.npz")
        
        assert ob.data.shape == (100,)
        
        os.unlink("test_observable.npz")
        
    def test_save_raw(self):
        
        data = random_complex(100)
        ob = Observable(data)
        ob.save_raw("test_observable.npy")
        
        assert os.path.exists("test_observable.npy")
        
        test_ob = np.load("test_observable.npy")
        
        assert (test_ob == data).all()
        
        os.unlink("test_observable.npy")

    def test_header(self):
        
        ob = Observable()
        header = ob.header()
        
        assert header == {}
        
    def test_addition(self):
        
        data1 = random_complex(100)
        data2 = random_complex(100)
        
        ob1 = Observable(data1)
        ob2 = Observable(data2)
        
        ob3 = ob1 + ob2
        
        assert (ob3.data == ob1.data + ob2.data).all()

    def test_division(self):
        
        data = random_complex(100)
        div = npr.random()
        
        ob1 = Observable(data)
        ob2 = ob1 / div
        
        assert (ob2.data == data / div).all()

    def test_negation(self):
        
        data = random_complex(100)
        
        ob1 = Observable(data)
        ob2 = -ob1
        
        assert (ob2.data == -data).all()
        
    def test_addition(self):
        
        data1 = random_complex(100)
        data2 = random_complex(100)
        
        ob1 = Observable(data1)
        ob2 = Observable(data2)
        
        ob3 = ob1 - ob2
        
        assert (ob3.data == ob1.data - ob2.data).all()

class TestConfig:
    
    def test_init(self):
        
        config_data = random_complex((4, 2, 2, 2, 4, 3, 3))
        
        with pytest.raises(ValueError):
            config = Config(config_data, 2, 2, 5.5, 1.0, 1.0, 1.0, "wilson")

        config = Config(config_data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson")
        
    def test_save(self):
        
        config_data = random_complex((4, 2, 2, 2, 4, 3, 3))
        
        config = Config(config_data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson")
        config.save("test_config.npz")
        
        assert os.path.exists("test_config.npz")

        test_config = np.load("test_config.npz")
        
        assert "data" in test_config.files
        assert "header" in test_config.files
        
        assert (test_config["data"] == config_data).all()
        assert test_config["header"] == {'L': 2, 'T': 4, 'beta': 5.5, 'ut': 1.0,
                                         'us': 1.0, 'chi': 1.0,
                                         'action': 'wilson'}

    def test_load(self):
        
        config = Config.load("test_config.npz")
        
        assert config.data.shape == (4, 2, 2, 2, 4, 3, 3)
        
        os.unlink("test_config.npz")

    def test_save_raw(self):
        
        config_data = random_complex((4, 2, 2, 2, 4, 3, 3))
        
        config = Config(config_data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson")
        config.save_raw("test_config.npy")
        
        assert os.path.exists("test_config.npy")
        
        test_config = np.load("test_config.npy")
        
        assert (test_config == config_data).all()
        
        os.unlink("test_config.npy")
        
    def test_header(self):
        
        config_data = random_complex((4, 2, 2, 2, 4, 3, 3))
        
        config = Config(config_data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson")
        header = config.header()
        
        assert header == {'L': 2, 'T': 4, 'beta': 5.5, 'ut': 1.0,
                          'us': 1.0, 'chi': 1.0, 'action': 'wilson'}

class TestPropagator:
    
    def test_init(self):
        
        data = random_complex((2, 2, 2, 2, 4, 4, 3, 3))
        
        with pytest.raises(ValueError):
            propagator = Propagator(data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson",
                                    "wilson", 0.4, {}, [0, 0, 0, 0], 0, 1.0,
                                    "jacobi", 0, 1.0, "jacobi", 0, 1.0)
        
        propagator = Propagator(data, 2, 2, 5.5, 1.0, 1.0, 1.0, "wilson",
                                "wilson", 0.4, {}, [0, 0, 0, 0], 0, 1.0,
                                "jacobi", 0, 1.0, "jacobi", 0, 1.0)

    def test_save(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson",
                          "wilson", 0.4, {}, [0, 0, 0, 0], 0, 1.0, "jacobi",
                          0, 1.0, "jacobi", 0, 1.0)
        prop.save("test_prop.npz")
        
        assert os.path.exists("test_prop.npz")

        test_prop = np.load("test_prop.npz")
        
        assert "data" in test_prop.files
        assert "header" in test_prop.files
        
        assert (test_prop["data"] == prop_data).all()
        assert test_prop["header"].item() == {'L': 2, 'T': 4, 'beta': 5.5,
                                              'ut': 1.0, 'us': 1.0, 'chi': 1.0,
                                              'gauge_action': 'wilson',
                                              'fermion_action': "wilson",
                                              'mass': 0.4,
                                              'action_parameters': {},
                                              'source_site': [0, 0, 0, 0],
                                              'num_field_smears': 0,
                                              'field_smearing_param': 1.0,
                                              'source_smearing_type': "jacobi",
                                              'num_source_smears': 0,
                                              'source_smearing_param': 1.0,
                                              'sink_smearing_type': "jacobi",
                                              'num_sink_smears': 0,
                                              'sink_smearing_param': 1.0}
    
    def test_load(self):
        
        prop = Propagator.load("test_prop.npz")
        
        assert prop.data.shape == (4, 2, 2, 2, 4, 4, 3, 3)
        
        os.unlink("test_prop.npz")

    def test_save_raw(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson",
                          "wilson", 0.4, {}, [0, 0, 0, 0], 0, 1.0, "jacobi",
                          0, 1.0, "jacobi", 0, 1.0)
        prop.save_raw("test_prop.npy")
        
        assert os.path.exists("test_prop.npy")
        
        test_prop = np.load("test_prop.npy")
        
        assert (test_prop == prop_data).all()
        
        os.unlink("test_prop.npy")
        
    def test_header(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson",
                          "wilson", 0.4, {}, [0, 0, 0, 0], 0, 1.0, "jacobi",
                          0, 1.0, "jacobi", 0, 1.0)
        header = prop.header()
        
        assert header == {'L': 2, 'T': 4, 'beta': 5.5,
                          'ut': 1.0, 'us': 1.0, 'chi': 1.0,
                          'gauge_action': 'wilson',
                          'fermion_action': 'wilson',
                          'mass': 0.4, 'action_parameters': {},
                          'source_site': [0, 0, 0, 0],
                          'num_field_smears': 0,
                          'field_smearing_param': 1.0,
                          'source_smearing_type': 'jacobi',
                          'num_source_smears': 0,
                          'source_smearing_param': 1.0,
                          'sink_smearing_type': 'jacobi',
                          'num_sink_smears': 0,
                          'sink_smearing_param': 1.0}
    
    def test_conjugate(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson",
                          "wilson", 0.4, {}, [0, 0, 0, 0], 0, 1.0, "jacobi",
                          0, 1.0, "jacobi", 0, 1.0)
        
        prop_conj = prop.conjugate()
        
        assert (prop_conj.data == np.conj(prop_data)).all()
        
    def test_transpose_spin(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson",
                          "wilson", 0.4, {}, [0, 0, 0, 0], 0, 1.0, "jacobi",
                          0, 1.0, "jacobi", 0, 1.0)
        
        prop_transpose = prop.transpose_spin()
        
        assert (prop_transpose.data == np.swapaxes(prop_data, 4, 5)).all()
        
    def test_transpose_colour(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson",
                          "wilson", 0.4, {}, [0, 0, 0, 0], 0, 1.0, "jacobi",
                          0, 1.0, "jacobi", 0, 1.0)
        
        prop_transpose = prop.transpose_colour()
        
        assert (prop_transpose.data == np.swapaxes(prop_data, 6, 7)).all()
        
    def test_adjoint(self):
        
        prop_data = np.load("{}/propagator_tree_level_4c8_4000_no_smear.npy"
                            .format(data_dir))
        prop = Propagator(prop_data, 4, 8, 5.5, 1.0, 1.0, 1.0, "wilson",
                          "wilson", 0.4, {}, [0, 0, 0, 0], 0, 1.0, "jacobi",
                          0, 1.0, "jacobi", 0, 1.0)
        
        prop_adjoint = prop.adjoint()
        
        expected_prop = np.conj(np.transpose(prop_data,
                                             [0, 1, 2, 3, 5, 4, 7, 6]))
        
        expected_prop = np.tensordot(expected_prop, gamma5, (5, 0))
        expected_prop = np.transpose(expected_prop, [0, 1, 2, 3, 4, 7, 5, 6])
        expected_prop = np.tensordot(gamma5, expected_prop, (1, 4))
        expected_prop = np.transpose(expected_prop, [1, 2, 3, 4, 0, 5, 6, 7])
        
        assert (prop_adjoint.data == expected_prop).all()
                
    def test_multiply(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson",
                          "wilson", 0.4, {}, [0, 0, 0, 0], 0, 1.0, "jacobi",
                          0, 1.0, "jacobi", 0, 1.0)
        
        with pytest.raises(ValueError):
            prop_multiplied = prop * []
            
        scalar_multiple = npr.random()
        prop_multiplied = prop * scalar_multiple
        
        assert (prop_multiplied.data == prop_data * scalar_multiple).all()
        
        matrix_multiple = np.matrix(npr.random((4, 4)))
        prop_multiplied = prop * matrix_multiple
        
        expected_product = np.tensordot(prop_data, matrix_multiple, (5, 0))
        expected_product = np.swapaxes(np.swapaxes(expected_product, 6, 7), 5, 6)
        
        assert (prop_multiplied.data == expected_product).all()
        
        matrix_multiple = np.matrix(npr.random((3, 3)))
        prop_multiplied = prop * matrix_multiple
        
        expected_product = np.tensordot(prop_data, matrix_multiple, (7, 0))
        
        assert (prop_multiplied.data == expected_product).all()

    def test_right_multiply(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, 1.0, 1.0, "wilson",
                          "wilson", 0.4, {}, [0, 0, 0, 0], 0, 1.0, "jacobi",
                          0, 1.0, "jacobi", 0, 1.0)
        
        with pytest.raises(ValueError):
            prop_multiplied = prop * []
            
        scalar_multiple = npr.random()
        prop_multiplied = prop * scalar_multiple
        
        assert (prop_multiplied.data == prop_data * scalar_multiple).all()
        
        matrix_multiple = np.matrix(npr.random((4, 4)))
        prop_multiplied = matrix_multiple * prop
        
        expected_product = np.tensordot(matrix_multiple, prop_data, (1, 4))
        expected_product = np.transpose(expected_product,
                                        (1, 2, 3, 4, 0, 5, 6, 7))
        
        assert (prop_multiplied.data == expected_product).all()
        
        matrix_multiple = np.matrix(npr.random((3, 3)))
        prop_multiplied = matrix_multiple * prop
        
        expected_product = np.tensordot(matrix_multiple, prop_data, (1, 6))
        expected_product = np.transpose(expected_product,
                                        (1, 2, 3, 4, 5, 6, 0, 7))
        
        assert (prop_multiplied.data == expected_product).all()

class TestTwoPoint:
    
    def test_init(self):
        
        twopoint = TwoPoint(8, 4)
        
    def test_save(self):
        
        twopoint = TwoPoint(8, 4)
        attribute_key = ("pion", (), (0, 0, 0), None, None)
        correlator = npr.random(10)
        twopoint.data[attribute_key] = correlator
              
        twopoint.save("test_twopoint.npz")
        
        assert os.path.exists("test_twopoint.npz")
        
        test_twopoint = np.load("test_twopoint.npz")
        
        assert "data" in test_twopoint.files
        assert (test_twopoint["data"].item()[attribute_key] == correlator).all()
        
        assert test_twopoint["header"].item() == {'L': 4, 'T': 8}
        
    def test_load(self):
        
        twopoint = TwoPoint.load("test_twopoint.npz")
        attribute_key = ("pion", (), (0, 0, 0), None, None)
        
        assert twopoint.data[attribute_key].shape == (10,)
        assert len(twopoint.data.keys()) == 1
        
        #os.unlink("test_twopoint.npz")
        
    def test_save_raw(self):
        
        twopoint = TwoPoint(4, 8)
        
        with pytest.raises(NotImplementedError):
            twopoint.save_raw("test")
            
    def test_available_interpolators(self):
        
        mesons = TwoPoint.available_interpolators()
        
        assert len(mesons) == 16
        
        for meson in mesons:
            assert len(meson) == 2
            
    def test_get_correlator(self):
        
        twopoint = TwoPoint(8, 4)
        
        source_sink_types = ["point", "wall", "smeared", "stochastic"]
        
        for i in xrange(10):
            label = random_variable_name()
            momentum = [npr.randint(4) for i in xrange(3)]
            masses = [round(npr.random(), 4)
                      for i in xrange(npr.randint(1, 5))]
            source_type = source_sink_types[npr.randint(len(source_sink_types))]
            sink_type = source_sink_types[npr.randint(len(source_sink_types))]
            
            correlator = npr.random(8)
            twopoint.data[(label, tuple(masses), tuple(momentum), source_type,
                           sink_type)] = correlator
        
            assert (twopoint.get_correlator(label,
                                            masses,
                                            momentum,
                                            source_type,
                                            sink_type) == correlator).all()
        
        all_correlators = twopoint.get_correlator()
            
        assert type(all_correlators) == dict
        assert type(all_correlators.keys()[0]) == tuple
        assert len(all_correlators.values()) == 10
            
    def test_add_correlator(self):
        
        twopoint = TwoPoint(8, 4)
        
        correlator = npr.random(8)
        
        label = "dummy_name"
        
        twopoint.add_correlator(correlator, label)
        twopoint.add_correlator(correlator, label, [0.1, 0.1])
        twopoint.add_correlator(correlator, label, [0.1, 0.5, 0.2], [1, 1, 1])
        
        assert (twopoint.data[(label, (), (0, 0, 0), None, None)] \
                == correlator).all()
        assert (twopoint.data[(label, (0.1, 0.1), (0, 0, 0), None, None)] \
                == correlator).all()
        assert (twopoint.data[(label, (0.1, 0.5, 0.2), (1, 1, 1), None, None)] \
                == correlator).all()
        
        with pytest.raises(ValueError):
            twopoint.add_correlator(npr.random(3), label, [0.1, 0.1])
            twopoint.add_correlator(npr.random((2, 2)), label, [0.1, 0.1])
            
        raw_correlator = random_complex((8, 4, 4, 4))
        sites = [[x, y, z]
                 for x in xrange(4)
                 for y in xrange(4)
                 for z in xrange(4)]
        
        prefactors = np.exp(1j * np.pi / 2 * np.dot(sites, [1, 0, 0]))
        correlator = np.dot(np.reshape(raw_correlator, (8, 64)), prefactors).real
        twopoint.add_correlator(raw_correlator, label, [], [1, 0, 0],
                                projected=False)
        assert (twopoint.data[(label, (), (1, 0, 0), None, None)]
                == correlator).all()
        
        prefactors = np.exp(1j * np.pi / 2 * np.dot(sites, [1, 1, 0]))
        correlator = np.dot(np.reshape(raw_correlator, (8, 64)), prefactors).real
        twopoint.add_correlator(raw_correlator, label, [], [1, 1, 0],
                                projected=False)
        assert (twopoint.data[(label, (), (1, 1, 0), None, None)]
                == correlator).all()
        
        with pytest.raises(ValueError):
            twopoint.add_correlator(npr.random((4, 2, 2, 1)), label, [0.1, 0.1],
                                    [1, 1, 0], False)
            twopoint.add_correlator(npr.random((4, 2, 2)), label, [0.1, 0.1],
                                    [1, 1, 0], False)
            
    def test_load_chroma_mesonspec(self):
        
        twopoint = TwoPoint(32, 16)
        
        twopoint \
          .load_chroma_mesonspec(create_fullpath("mesonspec_hh_6000_1.xml"))
                                       
        assert len(twopoint.data.keys()) == 8
            
    def test_load_chroma_hadspec(self):
        
        twopoint = TwoPoint(8, 4)
        
        twopoint.load_chroma_hadspec(create_fullpath("hadspec.dat.xml"))
                                       
        assert len(twopoint.data.keys()) == 95
            
    def test_load_chroma_hadspec_mesons(self):
        
        twopoint = TwoPoint(8, 4)
        filename = create_fullpath("hadspec.dat.xml")
        twopoint.load_chroma_hadspec_currents(filename)
                                       
        assert len(twopoint.data.keys()) == 64
            
    def test_load_chroma_hadspec_baryons(self):
        
        twopoint = TwoPoint(8, 4)
        filename = create_fullpath("hadspec.dat.xml")
        twopoint.load_chroma_hadspec_currents(filename)
                                       
        assert len(twopoint.data.keys()) == 20
            
    def test_load_chroma_hadspec_currents(self):
        
        twopoint = TwoPoint(8, 4)
        filename = create_fullpath("hadspec.dat.xml")
        twopoint.load_chroma_hadspec_currents(filename)
                                       
        assert len(twopoint.data.keys()) == 11
        
    def test_load_chroma_mres(self):
        
        twopoint = TwoPoint(32, 16)
        
        twopoint.load_chroma_mres(create_fullpath("hadspec.out.xml"))
        
        assert len(twopoint.data.keys()) == 10
            
    def test_load_ukhadron_meson_binary(self):
        
        twopoint = TwoPoint(32, 16)
        
        filename = create_fullpath("meson_m_0.45_m_0.45_Z2.280.bin")        
        twopoint \
          .load_ukhadron_meson_binary(filename, "big", (0.1, 0.1))
                                       
        assert len(twopoint.data.keys()) == 256
        
    def test_load_ukhadron_mres(self):
        
        twopoint = TwoPoint(32, 16)
        
        twopoint.load_ukhadron_mres("{}/mres_data.xml".format(data_dir), 0.1)
        
        assert len(twopoint.data.keys()) == 2
            
    def test_compute_meson_correlator(self):
        
        tolerance = 1e-6
        
        expected_correlator = np.array([7.37116795e-01,
                                        2.22850450e-02,
                                        1.21447706e-03,
                                        6.72126426e-05,
                                        7.87223375e-06,
                                        7.73382888e-05,
                                        1.32329206e-03,
                                        2.35567921e-02])
        
        filename \
          = create_fullpath("prop_wilson_conjugate_gradient_jacobi_0_0_0.npy")
        propagator_data = np.load(filename)
        
        propagator = Propagator(propagator_data, 4, 8, 5.5, 1.0, 1.0, 1.0,
                                "wilson", "wilson", 0.4, {}, [0, 0, 0, 0],
                                0, 1.0, "jacobi", 0, 1.0, "jacobi", 0, 1.0)
        
        momenta = [0, 0, 0]
        twopoint = TwoPoint(8, 4)
        source_interpolators = [gamma5, "g5", "pion"]
        sink_interpolators = source_interpolators
        label = "pion"
        
        for source_interpolator, sink_interpolator \
          in zip(source_interpolators, sink_interpolators):
            twopoint.compute_meson_correlator(propagator, propagator,
                                              source_interpolator,
                                              sink_interpolator,
                                              label, momenta,
                                              average_momenta=True)
        
            correlator_name = "{}_px{}_py{}_pz{}_M{}_M{}_point_point" \
              .format(label, momenta[0], momenta[1], momenta[2], 0.4, 0.4) \
              .replace(".", "p")
              
            correlator_key = (label, (0.4, 0.4), tuple(momenta),
                              "point", "point")
        
            assert np.allclose(twopoint.data[correlator_key],
                               expected_correlator)
            
        momenta = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
        filename \
          = create_fullpath("prop_free_8c16_m1.0_no_smear.npy")
        propagator_data = np.load(filename)
        
        propagator = Propagator(propagator_data, 8, 16, 5.5, 1.0, 1.0, 1.0,
                                "wilson", "wilson", 1.0, {}, [0, 0, 0, 0],
                                0, 1.0, "jacobi", 0, 1.0, "jacobi", 0, 1.0)
        
        filename \
          = create_fullpath("correlators_free_8c16_m1.0.npy")
        expected_correlators = np.load(filename)
        
        twopoint = TwoPoint(16, 8)
        source_interpolator = gamma5
        sink_interpolator = gamma5
        label = "another_pion"
        
        twopoint.compute_meson_correlator(propagator, propagator,
                                          source_interpolator, sink_interpolator,
                                          label, momenta, average_momenta=True)
        
        for i, momentum in enumerate(momenta):
              
            correlator_key = (label, (1.0, 1.0), tuple(momentum),
                              "point", "point")
        
            assert np.allclose(twopoint.data[correlator_key],
                               expected_correlators[i])
            
    def test_fit_correlator(self):
        
        twopoint = TwoPoint(16, 8)
        
        mass = npr.random()
        amplitude = 1000 * npr.random()
        expected_result = np.array([amplitude, mass])
        
        correlator = amplitude * np.exp(-mass * np.arange(twopoint.T))        
        twopoint.add_correlator(correlator, "test", [0.1, 0.1], [0, 0, 0],
                                "point", "point")
        
        fit_function \
          = lambda b, t, Ct, err: Ct - b[0] * np.exp(-b[1] * t)
        
        fit_result = twopoint.fit_correlator(fit_function, range(twopoint.T),
                                             [500, 1])
        fit_result = np.array(fit_result)
        assert np.allclose(fit_result, expected_result)
        
        fit_result = twopoint.fit_correlator(fit_function, [0, twopoint.T],
                                             [500, 1])
        fit_result = np.array(fit_result)
        assert np.allclose(fit_result, expected_result)
        
        fit_result = twopoint.fit_correlator(fit_function, range(twopoint.T),
                                             [500, 1], np.ones(twopoint.T),
                                             lambda x: x[1]**2)
        assert np.allclose(fit_result, mass**2)
            
    def test_compute_energy(self):
        
        twopoint = TwoPoint(16, 8)
        
        mass = npr.random()
        amplitude = 1000 * npr.random()
        expected_result = np.array([amplitude, mass])
        
        correlator = amplitude * np.exp(-mass * np.arange(twopoint.T)) \
          + amplitude * np.exp(-mass * (twopoint.T - np.arange(twopoint.T)))
        twopoint.add_correlator(correlator, "test", [0.1, 0.1], [0, 0, 0],
                                "point", "point")
        
        fitted_mass = twopoint.compute_energy(range(twopoint.T), [500, 1])
        assert np.allclose(fitted_mass, mass)
            
    def test_compute_square_energy(self):
        
        twopoint = TwoPoint(16, 8)
        
        mass = npr.random()
        amplitude = 1000 * npr.random()
        expected_result = np.array([amplitude, mass])
        
        correlator = amplitude * np.exp(-mass * np.arange(twopoint.T)) \
          + amplitude * np.exp(-mass * (twopoint.T - np.arange(twopoint.T)))
        twopoint.add_correlator(correlator, "test", [0.1, 0.1], [0, 0, 0],
                                "point", "point")
        
        fitted_mass = twopoint.compute_square_energy(range(twopoint.T), [500, 1])
        assert np.allclose(fitted_mass, mass**2)
        
    def test_compute_c_square(self):
        
        twopoint = TwoPoint(16, 8)
        
        amplitude = 1000 * npr.random()
        mass = npr.random()
        energy1 = 1 + npr.random()
        energy2 = 2 + npr.random()
        momenta = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
        
        expected_result \
          = np.array([(energy1**2 - mass**2) / (np.pi / 4)**2,
                      (energy2**2 - mass**2) / ((np.pi / 4)**2 * 2)])
        
        for m, p in zip([mass, energy1, energy2], momenta):
            correlator = amplitude * np.exp(-m * np.arange(twopoint.T)) \
              + amplitude * np.exp(-m * (twopoint.T - np.arange(twopoint.T)))
              
            twopoint.add_correlator(correlator, "test", [0.1, 0.1], p, "point",
                                    "point")
            
        actual_result = twopoint.compute_c_square(3 * [range(16)], [500, 1],
                                                  [[1, 0, 0], [1, 1, 0]])
        
        assert np.allclose(actual_result, expected_result)
        
    def test_compute_effmass(self):
        
        twopoint = TwoPoint(16, 8)
        
        correlator = npr.random(twopoint.T)
        expected_effmass = np.log(np.abs(correlator / np.roll(correlator, -1)))
        
        twopoint.add_correlator(correlator, "test", [0.1, 0.1], [0, 0, 0],
                                "point", "point")
        
        actual_effmass = twopoint.compute_effmass("test", [0.1, 0.1], [0, 0, 0],
                                                  "point", "point")
        
        assert np.allclose(actual_effmass, expected_effmass)
        
    def test_div(self):
        
        twopoint = TwoPoint(16, 8)
        
        correlator = npr.random(16)
        
        twopoint.add_correlator(correlator, "test", [0.1, 0.1], [0, 0, 0],
                                "point", "point")

        div = npr.random()        
        twopoint_div = twopoint / div
        
        expected_correlator = correlator / div
        actual_correlator = twopoint_div.get_correlator("test", [0.1, 0.1],
                                                        [0, 0, 0], "point",
                                                        "point")
        assert np.allclose(actual_correlator, expected_correlator)
        
    def test_neg(self):
        
        twopoint = TwoPoint(16, 8)
        
        correlator = npr.random(16)
        
        twopoint.add_correlator(correlator, "test", [0.1, 0.1], [0, 0, 0],
                                "point", "point")
        
        twopoint_neg = -twopoint
        
        expected_correlator = -correlator
        actual_correlator = twopoint_neg.get_correlator("test", [0.1, 0.1],
                                                        [0, 0, 0], "point",
                                                        "point")
        assert np.allclose(actual_correlator, expected_correlator)
        
    def test_sub(self):
        
        twopoint1 = TwoPoint(16, 8)
        twopoint2 = TwoPoint(16, 8)
        
        correlator1 = npr.random(twopoint1.T)
        correlator2 = npr.random(twopoint2.T)
        correlator3 = npr.random(twopoint1.T)
        correlator4 = npr.random(twopoint2.T)
        
        twopoint1.add_correlator(correlator1, "test", [0.1, 0.1], [0, 0, 0],
                                 "point", "point")
        twopoint1.add_correlator(correlator3, "test", [0.2, 0.5], [0, 0, 0],
                                 "point", "point")
        twopoint2.add_correlator(correlator2, "test", [0.1, 0.1], [0, 0, 0],
                                 "point", "point")
        twopoint2.add_correlator(correlator4, "test", [1.0, 0.1], [1, 0, 0],
                                 "point", "point")
        
        with pytest.raises(KeyError):
            twopoint3 = twopoint1 - twopoint2
        
        twopoint1 = TwoPoint(16, 8)
        twopoint2 = TwoPoint(16, 8)
        
        twopoint1.add_correlator(correlator1, "test", [0.1, 0.1], [0, 0, 0],
                                 "point", "point")
        twopoint2.add_correlator(correlator2, "test", [0.1, 0.1], [0, 0, 0],
                                 "point", "point")
        
        twopoint3 = twopoint1 - twopoint2
        
        expected_correlator = correlator1 - correlator2
        actual_correlator = twopoint3.get_correlator("test", [0.1, 0.1],
                                                     [0, 0, 0], "point",
                                                     "point")
        assert np.allclose(actual_correlator, expected_correlator)

class TestDataSet:
    
    def test_init(self):
        
        try:
            shutil.rmtree("pyQCDcache")
        except:
            pass
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
    def test_add_datum(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))
        
        try:
            zfile = zipfile.ZipFile("test_data.zip", 'r', zipfile.ZIP_DEFLATED,
                                    True)
        except RuntimeError:
            zfile = zipfile.ZipFile("test_data.zip", 'r', zipfile.ZIP_STORED,
                                    False)
            
        assert len(zfile.namelist()) == 101

        with pytest.raises(TypeError):
            dataset.add_datum({})
        
        for i in xrange(100):
            assert zfile.namelist().count("floatWrapper{}.npz".format(i)) == 1
            
        zfile.close()
            
    def test_get_datum(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))
            
        for j, i in enumerate(rand_floats):
            assert i == dataset.get_datum(j)
            
    def test_set_datum(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        dataset.add_datum(floatWrapper(3))
        with pytest.raises(NotImplementedError):
            dataset.set_datum(0, floatWrapper(4))
            
    def test_apply_function(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        dataset.add_datum(floatWrapper(3))
        with pytest.raises(NotImplementedError):
            dataset.apply_function(lambda x: x, [])
            
    def test_measure(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))
            
        data_mean = rand_floats.mean()
            
        assert dataset.measure(lambda x: x**2) == data_mean**2
        assert dataset.measure(lambda x, b: b * np.sin(x), args=[2]) == 2 * np.sin(data_mean)
        
    def test_statistics(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))
            
        data_mean = rand_floats.mean()
        
        data_std = rand_floats.std()
        
        statistics = dataset.statistics()
        
        assert statistics[0] == data_mean
        assert statistics[1] == data_std
        
    def test_generate_bootstrap_cache(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))
        
        dataset.generate_bootstrap_cache(10)
        
        assert len(dataset.cache.keys()) == 10
        
        with pytest.raises(ValueError):
            dataset.generate_bootstrap_cache(10, -1)
            
        dataset.generate_bootstrap_cache(10, 3)
        
    def test_bootstrap(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))
            
        data_mean = rand_floats.mean()
        
        rtol = 0.1
        
        bootstrap_mean, bootstrap_std = dataset.bootstrap(lambda x: x**2, 100,
                                                          use_cache=False)
        assert np.allclose(bootstrap_mean, data_mean**2, rtol)
        
        bootstrap_mean, bootstrap_std = dataset.bootstrap(lambda x: x**2, 100,
                                                          use_cache=False,
                                                          use_parallel=True)
        assert np.allclose(bootstrap_mean, data_mean**2, rtol)
        
        bootstrap_mean, bootstrap_std = dataset.bootstrap(lambda x: x**2, 100)
        assert np.allclose(bootstrap_mean, data_mean**2, rtol)
        
        bootstrap_mean_2, bootstrap_std_2 = dataset.bootstrap(lambda x: x**2, 100,
                                                              use_cache=True,
                                                              use_parallel=True)
        assert bootstrap_mean == bootstrap_mean_2
        assert bootstrap_std == bootstrap_std_2
            
        bootstrap_mean, bootstrap_std = dataset.bootstrap(lambda x: x**2, 10, 3,
                                                          use_cache=False)
        assert np.allclose(bootstrap_mean, data_mean**2, rtol)
            
    def test_jackknife_datum(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))        
        
        assert dataset.jackknife_datum(0) == rand_floats[1:].mean()
        
        with pytest.raises(KeyError):
            dataset.jackknife_datum(100)
            
        jackknife_result = dataset.jackknife_datum(0, 3)
        
    def test_generate_jackknife_cache(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))
        
        dataset.generate_jackknife_cache()
        
        assert len(dataset.cache.keys()) == 100
            
        dataset.generate_jackknife_cache(3)
        
    def test_jackknife(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))
            
        data_mean = rand_floats.mean()
        
        rtol = 0.001
        
        jackknife_mean, jackknife_std = dataset.jackknife(lambda x: x**2)        
        assert np.allclose(jackknife_mean, data_mean**2, rtol)
        old_mean = jackknife_mean
        
        jackknife_mean, jackknife_std = dataset.jackknife(lambda x: x**2,
                                                          use_cache=False)
        assert np.allclose(jackknife_mean, data_mean**2, rtol)
        assert old_mean == jackknife_mean
        
        jackknife_mean, jackknife_std = dataset.jackknife(lambda x: x**2,
                                                          use_parallel=True)
        assert np.allclose(jackknife_mean, data_mean**2, rtol)
        assert old_mean == jackknife_mean
        
        with pytest.raises(ZeroDivisionError):
            result = dataset.jackknife(lambda x: x**2, 0)
            
        result = dataset.jackknife(lambda x: x**2, 3)
        assert np.allclose(result[0], data_mean**2, rtol)
            
    def test_load(self):
        
        dataset = DataSet.load("test_data.zip")
        
        assert dataset.num_data == 100
        
    def test_utils(self):
        
        a = npr.random(10)
        b = npr.random(10)
        
        # First check addition
        assert DataSet._add_measurements(a.tolist(), b.tolist()) \
          == (a + b).tolist()
        assert DataSet._add_measurements(tuple(a.tolist()), tuple(b.tolist())) \
          == (a + b).tolist()
          
        a_dict = dict(zip(range(10), a.tolist()))
        b_dict = dict(zip(range(10), b.tolist()))
        
        assert DataSet._add_measurements(a_dict, b_dict) \
          == dict(zip(b_dict.keys(), (a + b).tolist()))
        
        # Now subtraction
        assert DataSet._sub_measurements(a.tolist(), b.tolist()) \
          == (a - b).tolist()
        assert DataSet._sub_measurements(tuple(a.tolist()), tuple(b.tolist())) \
          == (a - b).tolist()
        
        assert DataSet._sub_measurements(a_dict, b_dict) \
          == dict(zip(b_dict.keys(), (a - b).tolist()))
        
        # Now multiplication
        assert DataSet._mul_measurements(a.tolist(), b.tolist()) \
          == (a * b).tolist()
        assert DataSet._mul_measurements(tuple(a.tolist()), tuple(b.tolist())) \
          == (a * b).tolist()
        
        assert DataSet._mul_measurements(a_dict, b_dict) \
          == dict(zip(b_dict.keys(), (a * b).tolist()))
        
        # Now division
        div = npr.random()
        assert DataSet._div_measurements(a.tolist(), div) \
          == (a / div).tolist()
        assert DataSet._div_measurements(tuple(a.tolist()), div) \
          == (a / div).tolist()
        
        assert DataSet._div_measurements(a_dict, div) \
          == dict(zip(a_dict.keys(), (a / div).tolist()))
          
        # Now square root
        assert DataSet._sqrt_measurements(a.tolist()) \
          == np.sqrt(a).tolist()
        assert DataSet._sqrt_measurements(tuple(a.tolist())) \
          == np.sqrt(a).tolist()
        
        assert DataSet._sqrt_measurements(a_dict) \
          == dict(zip(a_dict.keys(), np.sqrt(a).tolist()))
            
    def test_iteration(self):
        
        test_data = DataSet.load("test_data.zip")
        data = [x for x in test_data]
        os.unlink("test_data.zip")
          
class TestWilsonLoops:
    
    def test_init(self):
        
        wilslp_data = npr.random((4, 8))
        
        wilslps = WilsonLoops(wilslp_data, 4, 8, 5.5, 1.0, 1.0, 1.0, "wilson", 0, 1.0)
        
        with pytest.raises(ValueError):
            wilslps = WilsonLoops(wilslp_data.T, 4, 8, 5.5, 1.0, 1.0, 1.0,
                                  "wilson", 0, 1.0)
            
    def test_lattice_spacing(self):
        
        expected_lattice_spacing \
          = np.array([0.31695984599258381, 0.62152983253471605])
        
        wilslp_data = np.load(create_fullpath("wilslps_no_smear.npy"))
        wilslps = WilsonLoops(wilslp_data, 4, 8, 5.5, 1.0, 1.0, 1.0,
                              "wilson", 0, 1.0)
        
        actual_lattice_spacing = np.array(wilslps.lattice_spacing())
        
        assert np.allclose(actual_lattice_spacing, expected_lattice_spacing)
        
    def test_pair_potential(self):
        
        expected_potential = np.array([-5.2410818817814314e-14,
                                       0.66603341142068573,
                                       1.2964758874025355,
                                       1.9389738652985116])
        
        wilslp_data = np.load(create_fullpath("wilslps_no_smear.npy"))
        wilslps = WilsonLoops(wilslp_data, 4, 8, 5.5, 1.0, 1.0, 1.0,
                              "wilson", 0, 1.0)
        
        actual_potential = wilslps.pair_potential()
        
        assert np.allclose(actual_potential, expected_potential)
    
if lattice_exists:
    class TestSimulation:
        
        def test_init(self):
            
            simulation = Simulation(100, 10, 250)
            simulation = Simulation(100, 10, 250, "heatbath", False,
                                    rand_seed=-1, verbosity=0)
        
        def test_create_lattice(self):
            
            simulation = Simulation(100, 10, 250)
            simulation.create_lattice(4, 8, "wilson", 5.5)
            simulation.create_lattice(4, 8, "wilson", 5.5, 1.0, 4)
            
        def test_load_ensemble(self):
            
            simulation = Simulation(100, 10, 250)
            
            with pytest.raises(AttributeError):
                simulation.load_ensemble("dummy")
                
            simulation.create_lattice(4, 8, "wilson", 5.5)
            
            with pytest.raises(AttributeError):
                simulation.load_ensemble(create_fullpath("4c8_ensemble.zip"))
                
            simulation = Simulation(3, 10, 100)
            simulation.create_lattice(8, 8, "wilson", 5.5)
            
            with pytest.raises(AttributeError):
                simulation.load_ensemble(create_fullpath("4c8_ensemble.zip"))
                
            simulation = Simulation(3, 10, 100)
            simulation.create_lattice(4, 16, "wilson", 5.5)
            
            with pytest.raises(AttributeError):
                simulation.load_ensemble(create_fullpath("4c8_ensemble.zip"))
                
            simulation = Simulation(3, 10, 100)
            simulation.create_lattice(4, 8, "wilson", 5.5)
            simulation.load_ensemble(create_fullpath("4c8_ensemble.zip"))
        
        def test_add_measurement(self):
        
            simulation = Simulation(100, 10, 250)
            simulation.add_measurement(Lattice.get_config, Config,
                                       "configs.zip")
            simulation.add_measurement(Lattice.get_config, Config, "configs.zip",
                                       meas_message="Getting correlator")
            
        def test_run(self):
            
            simulation = Simulation(5, 10, 100)
            simulation.create_lattice(4, 8, "wilson", 5.5)
            simulation.add_measurement(Lattice.get_config, Config, "configs.zip",
                                       meas_message="Storing gauge configuration")
            simulation.run()
        
            simulation = Simulation(3, 10, 100)
            simulation.create_lattice(4, 8, "wilson", 5.5)
            simulation.load_ensemble(create_fullpath("4c8_ensemble.zip"))
            simulation.add_measurement(Lattice.get_config, Config, "configs.zip")
            simulation.run()
