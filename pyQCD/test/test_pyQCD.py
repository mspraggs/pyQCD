import pytest
import numpy as np
import numpy.random as npr
import scipy.linalg as spla

import os
import itertools

from pyQCD import *
        
data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
        
def random_complex(shape):
    
    return npr.random(shape) * np.exp(2 * np.pi * 1j * npr.random(shape))

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
        
def test_random_su3_transform():
    
    lattice = Lattice()
    
    random_su3_transform(lattice)
    
    links = make_links(lattice.T, lattice.L)
    
    for link in links:
        determinant = spla.det(lattice.get_link(link))
        assert np.abs(determinant - 1.0) < 1e-12
        
        UUdagger = np.dot(lattice.get_link(link),
                          np.conj(lattice.get_link(link).T))
        assert (np.abs(UUdagger - np.identity(3))
                < 1e-12 * np.ones((3, 3))).all()
    
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
        assert config.u0 == lattice.u0
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
            data[t][x][y][z] = make_random_su(3)

        config = Config(data, lattice.L, lattice.T, lattice.beta, lattice.u0,
                        lattice.action)
        
        lattice.set_config(config)
        
        assert lattice.L == config.L
        assert lattice.T == config.T
        assert lattice.beta == config.beta
        assert lattice.u0 == config.u0
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
        
        lattice = Lattice(rand_seed = 0, update_method = "heatbath")
        tolerance = 1e-14 * np.ones((lattice.T, lattice.L, lattice.L,
                                     lattice.L, 4, 3, 3))
        lattice.update()
        
        expected_config = np.load("{}/config_heatbath_rs0.npy".format(data_dir))
        actual_config = lattice.get_config().data
        
        assert (np.abs(expected_config - actual_config) < tolerance).all()
        
        lattice = Lattice(rand_seed = 1, update_method = "heatbath")
        lattice.update()
        
        expected_config = np.load("{}/config_heatbath_rs1.npy".format(data_dir))
        actual_config = lattice.get_config().data
        
        assert (np.abs(expected_config - actual_config) < tolerance).all()
        
        lattice = Lattice(rand_seed = 0, update_method = "metropolis")
        lattice.update()
        
        expected_config = np.load("{}/config_metropolis_rs0.npy".format(data_dir))
        actual_config = lattice.get_config().data
        
        assert (np.abs(expected_config - actual_config) < tolerance).all()
        
        lattice = Lattice(rand_seed = 1, update_method = "metropolis")
        lattice.update()
        
        expected_config = np.load("{}/config_metropolis_rs1.npy".format(data_dir))
        actual_config = lattice.get_config().data
        
        assert (np.abs(expected_config - actual_config) < tolerance).all()
        
        lattice = Lattice(rand_seed = 0, update_method = "staple_metropolis")
        lattice.update()
        
        expected_config = np.load("{}/config_staple_metropolis_rs0.npy".format(data_dir))
        actual_config = lattice.get_config().data
        
        assert (np.abs(expected_config - actual_config) < tolerance).all()
        
        lattice = Lattice(rand_seed = 1, update_method = "staple_metropolis")
        lattice.update()
        
        expected_config = np.load("{}/config_staple_metropolis_rs1.npy".format(data_dir))
        actual_config = lattice.get_config().data
        
        assert (np.abs(expected_config - actual_config) < tolerance).all()      

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
        
        assert np.abs(lattice.get_av_plaquette() - 0.5) < 0.1

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
            assert np.abs(P1 - P2) < 5e-15

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
            assert np.abs(R1 - R2) < 5e-15

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
            assert np.abs(T1 - T2) < 5e-15

    def test_get_wilson_loop(self):
        
        lattice = Lattice()
        
        sites = make_sites(lattice.T, lattice.L)
        
        T_range = [1, lattice.T / 2, lattice.T - 1]
        L_range = [1, lattice.L / 2, lattice.L - 1]
        
        for site in sites:
            for r, t in itertools.product(L_range, T_range):
                for dim in range(1, 4):
                    for n in range(3):
                        assert lattice.get_wilson_loop(site, r, t, dim, n, 0.5) \
                          == 1.0
                
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
            assert np.abs(W1 - W2) < 5e-15
        
        W1 = lattice.get_wilson_loop([0, 0, 0, 0, 0], 4, 4, 1)
        for n in xrange(10):
            W2 = lattice.get_wilson_loop([0, 0, 0, 0, 0], 4, 4, 1, 0, 0.1 * n)
            assert W1 == W2

    def test_get_av_plaquette(self):
        
        lattice = Lattice(rand_seed=0)
        lattice.update()
        
        assert np.abs(lattice.get_av_plaquette() - 0.6744055385048071) < 1e-12
        
        random_su3_transform(lattice)
        
        assert np.abs(lattice.get_av_plaquette() - 0.6744055385048071) < 1e-12

    def test_get_av_rectangle(self):
        
        lattice = Lattice(rand_seed=0)
        lattice.update()
        
        assert np.abs(lattice.get_av_rectangle() - 0.5093032901600738) < 1e-12
        
        random_su3_transform(lattice)
        
        assert np.abs(lattice.get_av_rectangle() - 0.5093032901600738) < 1e-12

    def test_get_av_wilson_loop(self):
        
        lattice = Lattice(rand_seed=0)
        lattice.update()
        
        W = lattice.get_av_wilson_loop(4, 4)
        assert np.abs(W - 0.2883925516552541) < 1e-12

        random_su3_transform(lattice)
        
        W = lattice.get_av_wilson_loop(4, 4)
        assert np.abs(W - 0.2883925516552541) < 1e-12

    def test_get_wilson_loops(self):
        
        lattice = Lattice()
        
        for n in xrange(3):
            wilson_loops = lattice.get_wilson_loops(n, 0.5)
        
            for r in xrange(lattice.L):
                for t in xrange(lattice.T):
                    assert wilson_loops.data[r, t] \
                      == lattice.get_av_wilson_loop(r, t, n, 0.5)

    def test_get_propagator(self):
        
        lattice = Lattice(rand_seed=0, update_method="heatbath")
        
        propagator = lattice.get_propagator(0.4)
        
        expected_shape = (lattice.T, lattice.L, lattice.L, lattice.L, 4, 4, 3, 3)
        actual_shape = propagator.data.shape
        
        assert expected_shape == actual_shape
        
        tolerance = 1e-12 * np.ones(expected_shape)
        
        expected_propagator \
          = np.load("{}/propagator_tree_level_4c8_4000_no_smear.npy"
                    .format(data_dir))
        actual_propagator = propagator.data
        
        assert (np.abs(expected_propagator - actual_propagator)
                < tolerance).all()
        
        lattice.update()
        
        expected_propagator \
          = np.load("{}/propagator_tree_level_4c8_4000_no_smear.npy"
                    .format(data_dir))
        actual_propagator = propagator.data
        
        assert (np.abs(expected_propagator - actual_propagator)
                < tolerance).all()        
    
    def test_get_av_link(self):
        
        lattice = Lattice()

        assert lattice.get_av_link() - 1.0 < 1e-10

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
        
        config_data = random.complex((4, 2, 2, 2, 4, 3, 3))
        
        with pytest.raises(ValueError):
            config = Config(config_data, 2, 2, 5.5, 1.0, "wilson")

        config = Config(config_data, 2, 4, 5.5, 1.0, "wilson")
        
    def test_save(self):
        
        config_data = random_complex((4, 2, 2, 2, 4, 3, 3))
        
        config = Config(config_data, 2, 4, 5.5, 1.0, "wilson")
        config.save("test_config.npz")
        
        assert os.path.exists("test_config.npz")

        test_config = np.load("test_config.npz")
        
        assert "data" in test_config.files
        assert "header" in test_config.files
        
        assert (test_config["data"] == config_data).all()
        assert test_config["header"] == {'L': 2, 'T': 4, 'beta': 5.5, 'u0': 1.0,
                                         'action': 'wilson'}

    def test_load(self):
        
        config = Config.load("test_config.npz")
        
        assert config.data.shape == (4, 2, 2, 2, 4, 3, 3)
        
        os.unlink("test_config.npz")

    def test_save_raw(self):
        
        config_data = random.complex((4, 2, 2, 2, 4, 3, 3))
        
        config = Config(config_data, 2, 4, 5.5, 1.0, "wilson")
        config.save_raw("test_config.npy")
        
        assert os.path.exists("test_config.npy")
        
        test_config = np.load("test_config.npy")
        
        assert (test_config == config_data).all()
        
        os.unlink("test_config.npy")
        
    def test_header(self):
        
        config_data = random_complex((4, 2, 2, 2, 4, 3, 3))
        
        config = Config(config_data, 2, 4, 5.5, 1.0, "wilson")
        header = config.header()
        
        assert header == {'L': 2, 'T': 4, 'beta': 5.5, 'u0': 1.0,
                          'action': 'wilson'}
