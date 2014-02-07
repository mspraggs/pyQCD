import pytest
import numpy as np
import numpy.random as npr
import scipy.linalg as spla

import os
import shutil
import itertools
import string
import zipfile

from pyQCD import *
        
data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
        
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
        
        config_data = random_complex((4, 2, 2, 2, 4, 3, 3))
        
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
        
        config_data = random_complex((4, 2, 2, 2, 4, 3, 3))
        
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

class TestPropagator:
    
    def test_init(self):
        
        data = random_complex((2, 2, 2, 2, 4, 4, 3, 3))
        
        with pytest.raises(ValueError):
            propagator = Propagator(data, 2, 4, 5.5, 1.0, "wilson", 0.4,
                                    [0, 0, 0, 0], 0, 1.0, 0, 1.0, 0, 1.0)
        
        propagator = Propagator(data, 2, 2, 5.5, 1.0, "wilson", 0.4,
                                [0, 0, 0, 0], 0, 1.0, 0, 1.0, 0, 1.0)

    def test_save(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, "wilson", 0.4, [0, 0, 0, 0],
                          0, 1.0, 0, 1.0, 0, 1.0)
        prop.save("test_prop.npz")
        
        assert os.path.exists("test_prop.npz")

        test_prop = np.load("test_prop.npz")
        
        assert "data" in test_prop.files
        assert "header" in test_prop.files
        
        assert (test_prop["data"] == prop_data).all()
        assert test_prop["header"].item() == {'L': 2, 'T': 4, 'beta': 5.5,
                                              'u0': 1.0, 'action': 'wilson',
                                              'mass': 0.4,
                                              'source_site': [0, 0, 0, 0],
                                              'num_field_smears': 0,
                                              'field_smearing_param': 1.0,
                                              'num_source_smears': 0,
                                              'source_smearing_param': 1.0,
                                              'num_sink_smears': 0,
                                              'sink_smearing_param': 1.0}
    
    def test_load(self):
        
        prop = Propagator.load("test_prop.npz")
        
        assert prop.data.shape == (4, 2, 2, 2, 4, 4, 3, 3)
        
        os.unlink("test_prop.npz")

    def test_save_raw(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, "wilson", 0.4, [0, 0, 0, 0],
                          0, 1.0, 0, 1.0, 0, 1.0)
        prop.save_raw("test_prop.npy")
        
        assert os.path.exists("test_prop.npy")
        
        test_prop = np.load("test_prop.npy")
        
        assert (test_prop == prop_data).all()
        
        os.unlink("test_prop.npy")
        
    def test_header(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, "wilson", 0.4, [0, 0, 0, 0],
                          0, 1.0, 0, 1.0, 0, 1.0)
        header = prop.header()
        
        assert header == {'L': 2, 'T': 4, 'beta': 5.5,
                          'u0': 1.0, 'action': 'wilson',
                          'mass': 0.4,
                          'source_site': [0, 0, 0, 0],
                          'num_field_smears': 0,
                          'field_smearing_param': 1.0,
                          'num_source_smears': 0,
                          'source_smearing_param': 1.0,
                          'num_sink_smears': 0,
                          'sink_smearing_param': 1.0}
    
    def test_conjugate(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, "wilson", 0.4, [0, 0, 0, 0],
                          0, 1.0, 0, 1.0, 0, 1.0)
        
        prop_conj = prop.conjugate()
        
        assert (prop_conj.data == np.conj(prop_data)).all()
        
    def test_transpose_spin(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, "wilson", 0.4, [0, 0, 0, 0],
                          0, 1.0, 0, 1.0, 0, 1.0)
        
        prop_transpose = prop.transpose_spin()
        
        assert (prop_transpose.data == np.swapaxes(prop_data, 4, 5)).all()
        
    def test_transpose_colour(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, "wilson", 0.4, [0, 0, 0, 0],
                          0, 1.0, 0, 1.0, 0, 1.0)
        
        prop_transpose = prop.transpose_colour()
        
        assert (prop_transpose.data == np.swapaxes(prop_data, 6, 7)).all()
        
    def test_adjoint(self):
        
        prop_data = np.load("{}/propagator_tree_level_4c8_4000_no_smear.npy"
                            .format(data_dir))
        prop = Propagator(prop_data, 4, 8, 5.5, 1.0, "wilson", 0.4, [0, 0, 0, 0],
                          0, 1.0, 0, 1.0, 0, 1.0)
        
        prop_adjoint = prop.adjoint()
        
        expected_prop = np.conj(np.transpose(prop_data,
                                             [0, 1, 2, 3, 5, 4, 7, 6]))
        
        expected_prop = np.tensordot(expected_prop, constants.gamma5, (5, 0))
        expected_prop = np.transpose(expected_prop, [0, 1, 2, 3, 4, 7, 5, 6])
        expected_prop = np.tensordot(constants.gamma5, expected_prop, (1, 4))
        expected_prop = np.transpose(expected_prop, [1, 2, 3, 4, 0, 5, 6, 7])
        
        assert (prop_adjoint.data == expected_prop).all()
                
    def test_multiply(self):
        
        prop_data = random_complex((4, 2, 2, 2, 4, 4, 3, 3))
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, "wilson", 0.4, [0, 0, 0, 0],
                          0, 1.0, 0, 1.0, 0, 1.0)
        
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
        prop = Propagator(prop_data, 2, 4, 5.5, 1.0, "wilson", 0.4, [0, 0, 0, 0],
                          0, 1.0, 0, 1.0, 0, 1.0)
        
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
        attribute_name = "pion_px0_py0_pz0"
        correlator = npr.random(10)
        setattr(twopoint, attribute_name, correlator)
        twopoint.computed_correlators.append(attribute_name)
              
        twopoint.save("test_twopoint.npz")
        
        assert os.path.exists("test_twopoint.npz")
        
        test_twopoint = np.load("test_twopoint.npz")
        
        assert attribute_name in test_twopoint.files
        assert (test_twopoint[attribute_name] == correlator).all()
        
        assert test_twopoint["header"].item() == {'L': 4, 'T': 8,
                                                  'computed_correlators':
                                                  ['pion_px0_py0_pz0']}
        
    def test_load(self):
        
        twopoint = TwoPoint.load("test_twopoint.npz")
        
        assert twopoint.pion_px0_py0_pz0.shape == (10,)
        assert len(twopoint.computed_correlators) == 1
        
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
            masses = [npr.random() for i in xrange(npr.randint(1, 5))]
            source_type = source_sink_types[npr.randint(len(source_sink_types))]
            sink_type = source_sink_types[npr.randint(len(source_sink_types))]
            
            attribute_name \
              = "{0}_px{1}_py{2}_pz{3}".format(label, *momentum)
              
            for mass in masses:
                attribute_name \
                  += "_M{0}".format(round(mass, 4)).replace(".", "p")
                
            attribute_name += "_{0}_{1}".format(source_type, sink_type)
            
            correlator = npr.random(8)
            setattr(twopoint, attribute_name, correlator)
            twopoint.computed_correlators.append(attribute_name)
        
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
        
        assert (twopoint.dummy_name_px0_py0_pz0_None_None \
                == correlator).all()
        assert (twopoint.dummy_name_px0_py0_pz0_M0p1_M0p1_None_None \
                == correlator).all()
        assert (twopoint.dummy_name_px1_py1_pz1_M0p1_M0p5_M0p2_None_None \
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
        assert (twopoint.dummy_name_px1_py0_pz0_None_None == correlator).all()
        
        prefactors = np.exp(1j * np.pi / 2 * np.dot(sites, [1, 1, 0]))
        correlator = np.dot(np.reshape(raw_correlator, (8, 64)), prefactors).real
        twopoint.add_correlator(raw_correlator, label, [], [1, 1, 0],
                                projected=False)
        assert (twopoint.dummy_name_px1_py1_pz0_None_None == correlator).all()
        
        with pytest.raises(ValueError):
            twopoint.add_correlator(npr.random((4, 2, 2, 1)), label, [0.1, 0.1],
                                    [1, 1, 0], False)
            twopoint.add_correlator(npr.random((4, 2, 2)), label, [0.1, 0.1],
                                    [1, 1, 0], False)
            
    def test_load_chroma_mesonspec(self):
        
        twopoint = TwoPoint(32, 16)
        
        twopoint.load_chroma_mesonspec("{}/mesonspec_hh_6000_1.xml" \
                                       .format(data_dir))
                                       
        assert len(twopoint.computed_correlators) == 8
            
    def test_load_chroma_hadspec_mesons(self):
        
        twopoint = TwoPoint(8, 4)
        
        twopoint.load_chroma_hadspec_mesons("{}/hadspec.dat.xml" \
                                            .format(data_dir))
                                       
        assert len(twopoint.computed_correlators) == 64
            
    def test_load_chroma_hadspec_baryons(self):
        
        twopoint = TwoPoint(8, 4)
        
        twopoint.load_chroma_hadspec_baryons("{}/hadspec.dat.xml" \
                                             .format(data_dir))
                                       
        assert len(twopoint.computed_correlators) == 20
            
    def test_compute_meson_correlator(self):
        
        tolerance = 1e-6
        
        expected_correlator = np.array([0.7499591307765168,
                                        0.0475944363680059,
                                        0.013851732183308966,
                                        0.007241421784254808,
                                        0.0057323242970728815,
                                        0.00724142178425481,
                                        0.013851732183308969,
                                        0.047594436368005914])
        
        propagator_data \
          = np.load("{}/propagator_tree_level_4c8_4000_no_smear.npy"
                    .format(data_dir))
        
        propagator = Propagator(propagator_data, 4, 8, 5.5, 1.0, "wilson", 0.4,
                                [0, 0, 0, 0], 0, 1.0, 0, 1.0, 0, 1.0)
        
        momenta = [0, 0, 0]
        twopoint = TwoPoint(8, 4)
        source_interpolators = [constants.gamma5, "g5", "pion"]
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
        
            assert (np.abs(getattr(twopoint, correlator_name)
                           - expected_correlator)
                    < np.abs(expected_correlator)).all()
            
        momenta = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
        
        propagator_data \
          = np.load("{}/propagator_tree_level_8c16_10000_no_smear.npy"
                    .format(data_dir))
        
        propagator = Propagator(propagator_data, 8, 16, 5.5, 1.0, "wilson", 1.0,
                                [0, 0, 0, 0], 0, 1.0, 0, 1.0, 0, 1.0)
        
        expected_correlators \
          = np.load("{}/correlators_tree_level_8c16_10000.npy"
                    .format(data_dir))
        
        twopoint = TwoPoint(16, 8)
        source_interpolator = constants.gamma5
        sink_interpolator = constants.gamma5
        label = "another_pion"
        
        twopoint.compute_meson_correlator(propagator, propagator,
                                          source_interpolator, sink_interpolator,
                                          label, momenta, average_momenta=True)
        
        for i, momentum in enumerate(momenta):
            
            correlator_name = "{}_px{}_py{}_pz{}_M{}_M{}_point_point" \
              .format(label, momentum[0], momentum[1], momentum[2], 1.0, 1.0) \
              .replace(".", "p")
        
            assert (np.abs(getattr(twopoint, correlator_name)
                           - expected_correlators[i])
                    < np.abs(expected_correlator[i])).all()
            
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
        assert (np.abs(fit_result - expected_result)
                < 1e-6 * np.abs(expected_result)).all()
        
        fit_result = twopoint.fit_correlator(fit_function, [0, twopoint.T],
                                             [500, 1])
        fit_result = np.array(fit_result)
        assert (np.abs(fit_result - expected_result)
                < 1e-6 * np.abs(expected_result)).all()
        
        fit_result = twopoint.fit_correlator(fit_function, range(twopoint.T),
                                             [500, 1], np.ones(twopoint.T),
                                             lambda x: x[1]**2)
        assert np.abs(fit_result - mass**2) < 1e-6 * mass**2
            
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
        assert np.abs(fitted_mass - mass) < 1e-6 * mass
            
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
        assert np.abs(fitted_mass - mass**2) < 1e-6 * mass**2
        
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
        
        assert (np.abs(actual_result - expected_result)
                < 1e-6 * np.abs(expected_result)).all()
        
    def test_compute_effmass(self):
        
        twopoint = TwoPoint(16, 8)
        
        correlator = npr.random(twopoint.T)
        expected_effmass = np.log(np.abs(correlator / np.roll(correlator, -1)))
        
        twopoint.add_correlator(correlator, "test", [0.1, 0.1], [0, 0, 0],
                                "point", "point")
        
        actual_effmass = twopoint.compute_effmass("test", [0.1, 0.1], [0, 0, 0],
                                                  "point", "point")
        
        assert (np.abs(actual_effmass - expected_effmass)
                < 1e-6 * np.abs(expected_effmass)).all()
        
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
        assert (np.abs(actual_correlator - expected_correlator)
                < 1e-10 * np.abs(expected_correlator)).all()
        
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
        assert (np.abs(actual_correlator - expected_correlator)
                < 1e-10 * np.abs(expected_correlator)).all()
        
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
        
        twopoint3 = twopoint1 - twopoint2
        
        expected_correlator = correlator1 - correlator2
        actual_correlator = twopoint3.get_correlator("test", [0.1, 0.1],
                                                     [0, 0, 0], "point",
                                                     "point")
        assert (np.abs(actual_correlator - expected_correlator)
                < 1e-10 * np.abs(expected_correlator)).all()
        
        expected_correlator = correlator3
        actual_correlator = twopoint3.get_correlator("test", [0.2, 0.5],
                                                     [0, 0, 0], "point",
                                                     "point")
        assert (np.abs(actual_correlator - expected_correlator)
                < 1e-10 * np.abs(expected_correlator)).all()
        
        expected_correlator = -correlator4
        actual_correlator = twopoint3.get_correlator("test", [1.0, 0.1],
                                                     [1, 0, 0], "point",
                                                     "point")
        assert (np.abs(actual_correlator - expected_correlator)
                < 1e-10 * np.abs(expected_correlator)).all()

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
        
        for i in xrange(100):
            assert zfile.namelist().count("floatWrapper{}.npz".format(i)) == 1
        
        with pytest.raises(TypeError):
            dataset.add_datum({})
            
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
        
        files = os.listdir("pyQCDcache")
        
        assert len(files) == 10
        
        with pytest.raises(ValueError):
            dataset.generate_bootstrap_cache(10, -1)
            
        dataset.generate_bootstrap_cache(10, 3)
        
        shutil.rmtree("pyQCDcache")
        
    def test_bootstrap(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))
            
        data_mean = rand_floats.mean()
        
        bootstrap_mean, bootstrap_std = dataset.bootstrap(lambda x: x**2, 100)
        assert np.abs(bootstrap_mean - data_mean**2) < 0.1 * data_mean**2
        
        bootstrap_mean, bootstrap_std = dataset.bootstrap(lambda x: x**2, 100,
                                                          use_cache=False)
        assert np.abs(bootstrap_mean - data_mean**2) < 0.1 * data_mean**2
        
        with pytest.raises(ValueError):
            result = dataset.bootstrap(lambda x: x**2, 10, -1)
            
        result = dataset.bootstrap(lambda x: x**2, 10, 3, use_cache=False)
            
        dataset.generate_bootstrap_cache(10, 3)
        
        shutil.rmtree("pyQCDcache")
            
    def test_jackknife_datum(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))        
        
        assert dataset.jackknife_datum(0) == rand_floats[1:].mean()
        
        with pytest.raises(ValueError):
            dataset.jackknife_datum(0, -1)
            dataset.jackknife_datum(100)
            
        jackknife_result = dataset.jackknife_datum(0, 3)
        
    def test_generate_jackknife_cache(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))
        
        dataset.generate_jackknife_cache()
        
        files = os.listdir("pyQCDcache")
        assert len(files) == 100
        
        with pytest.raises(ValueError):
            dataset.generate_jackknife_cache(-1)
            
        dataset.generate_jackknife_cache(3)
        
        shutil.rmtree("pyQCDcache")
        
    def test_jackknife(self):
        
        dataset = DataSet(floatWrapper, "test_data.zip")
        
        rand_floats = npr.randint(100, size=100)
        
        for i in rand_floats:
            dataset.add_datum(floatWrapper(i))
            
        data_mean = rand_floats.mean()
        
        jackknife_mean, jackknife_std = dataset.jackknife(lambda x: x**2)        
        assert np.abs(jackknife_mean - data_mean**2) < 0.001 * data_mean**2
        
        jackknife_mean, jackknife_std = dataset.jackknife(lambda x: x**2,
                                                          use_cache=False) 
        assert np.abs(jackknife_mean - data_mean**2) < 0.001 * data_mean**2
        
        with pytest.raises(ValueError):
            result = dataset.jackknife(lambda x: x**2, 0)
            
        result = dataset.jackknife(lambda x: x**2, 3)
        
        shutil.rmtree("pyQCDcache")
            
    def test_load(self):
        
        dataset = DataSet.load("test_data.zip")
        
        assert dataset.num_data == 100
        os.unlink("test_data.zip")
        
    def test_utils(self):
        
        a = npr.random(10)
        b = npr.random(10)
        
        # First check addition
        assert DataSet._add_measurements(a.tolist(), b.tolist()) \
          == (a + b).tolist()
        assert DataSet._add_measurements(tuple(a.tolist()), tuple(b.tolist())) \
          == (a + b).tolist()
          
        a_dict = dict(zip(range(10), a.tolist()))
        assert DataSet._add_measurements(a_dict, tuple(b.tolist())) \
          == dict(zip(a_dict.keys(), (a + b).tolist()))
          
        b_dict = dict(zip(range(10), b.tolist()))
        assert DataSet._add_measurements(tuple(a.tolist()), b_dict) \
          == dict(zip(b_dict.keys(), (a + b).tolist()))
          
        b_dict = dict(zip(range(10), b.tolist()))
        assert DataSet._add_measurements(a_dict, b_dict) \
          == dict(zip(b_dict.keys(), (a + b).tolist()))
        
        # Now subtraction
        assert DataSet._sub_measurements(a.tolist(), b.tolist()) \
          == (a - b).tolist()
        assert DataSet._sub_measurements(tuple(a.tolist()), tuple(b.tolist())) \
          == (a - b).tolist()
          
        a_dict = dict(zip(range(10), a.tolist()))
        assert DataSet._sub_measurements(a_dict, tuple(b.tolist())) \
          == dict(zip(a_dict.keys(), (a - b).tolist()))
          
        b_dict = dict(zip(range(10), b.tolist()))
        assert DataSet._sub_measurements(tuple(a.tolist()), b_dict) \
          == dict(zip(b_dict.keys(), (a - b).tolist()))
          
        b_dict = dict(zip(range(10), b.tolist()))
        assert DataSet._sub_measurements(a_dict, b_dict) \
          == dict(zip(b_dict.keys(), (a - b).tolist()))
        
        # Now multiplication
        assert DataSet._mul_measurements(a.tolist(), b.tolist()) \
          == (a * b).tolist()
        assert DataSet._mul_measurements(tuple(a.tolist()), tuple(b.tolist())) \
          == (a * b).tolist()
          
        a_dict = dict(zip(range(10), a.tolist()))
        assert DataSet._mul_measurements(a_dict, tuple(b.tolist())) \
          == dict(zip(a_dict.keys(), (a * b).tolist()))
          
        b_dict = dict(zip(range(10), b.tolist()))
        assert DataSet._mul_measurements(tuple(a.tolist()), b_dict) \
          == dict(zip(b_dict.keys(), (a * b).tolist()))
          
        b_dict = dict(zip(range(10), b.tolist()))
        assert DataSet._mul_measurements(a_dict, b_dict) \
          == dict(zip(b_dict.keys(), (a * b).tolist()))
        
        # Now division
        div = npr.random()
        assert DataSet._div_measurements(a.tolist(), div) \
          == (a / div).tolist()
        assert DataSet._div_measurements(tuple(a.tolist()), div) \
          == (a / div).tolist()
          
        a_dict = dict(zip(range(10), a.tolist()))
        assert DataSet._div_measurements(a_dict, div) \
          == dict(zip(a_dict.keys(), (a / div).tolist()))
          
        # Now square root
        assert DataSet._sqrt_measurements(a.tolist()) \
          == np.sqrt(a).tolist()
        assert DataSet._sqrt_measurements(tuple(a.tolist())) \
          == np.sqrt(a).tolist()
          
        a_dict = dict(zip(range(10), a.tolist()))
        assert DataSet._sqrt_measurements(a_dict) \
          == dict(zip(a_dict.keys(), np.sqrt(a).tolist()))
          
class TestWilsonLoops:
    
    def test_init(self):
        
        wilslp_data = npr.random((4, 8))
        
        wilslps = WilsonLoops(wilslp_data, 4, 8, 5.5, 1.0, "wilson", 0, 1.0)
        
        with pytest.raises(ValueError):
            wilslps = WilsonLoops(wilslp_data.T, 4, 8, 5.5, 1.0, "wilson", 0,
                                  1.0)
            
    def test_lattice_spacing(self):
        
        expected_lattice_spacing \
          = np.array([0.31695984599258381, 0.62152983253471605])
        
        wilslp_data = np.load("{}/wilslps_no_smear.npy".format(data_dir))
        wilslps = WilsonLoops(wilslp_data, 4, 8, 5.5, 1.0, "wilson", 0, 1.0)
        
        actual_lattice_spacing = np.array(wilslps.lattice_spacing())
        
        assert (np.abs(actual_lattice_spacing - expected_lattice_spacing)
                < 1e-10 * np.abs(expected_lattice_spacing)).all()
        
    def test_pair_potential(self):
        
        expected_potential = np.array([-5.2410818817814314e-14,
                                       0.66603341142068573,
                                       1.2964758874025355,
                                       1.9389738652985116])
        
        wilslp_data = np.load("{}/wilslps_no_smear.npy".format(data_dir))
        wilslps = WilsonLoops(wilslp_data, 4, 8, 5.5, 1.0, "wilson", 0, 1.0)
        
        actual_potential = wilslps.pair_potential()
        
        assert (np.abs(actual_potential - expected_potential)
                < 1e-10 * np.abs(expected_potential)).all()
        
class TestSimulation:
    
    def test_init(self):
        
        simulation = Simulation(100, 10, 250)
        simulation = Simulation(100, 10, 250, "heatbath", False, rand_seed=-1,
                                verbosity=0)
        
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
            simulation.load_ensemble("{}/4c8_ensemble.zip".format(data_dir))
            
        simulation = Simulation(3, 10, 100)
        simulation.create_lattice(8, 8, "wilson", 5.5)
        
        with pytest.raises(AttributeError):
            simulation.load_ensemble("{}/4c8_ensemble.zip".format(data_dir))
            
        simulation = Simulation(3, 10, 100)
        simulation.create_lattice(4, 16, "wilson", 5.5)
        
        with pytest.raises(AttributeError):
            simulation.load_ensemble("{}/4c8_ensemble.zip".format(data_dir))
            
        simulation = Simulation(3, 10, 100)
        simulation.create_lattice(4, 8, "wilson", 5.5)
        simulation.load_ensemble("{}/4c8_ensemble.zip".format(data_dir))
        
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
        simulation.load_ensemble("{}/4c8_ensemble.zip".format(data_dir))
        simulation.add_measurement(Lattice.get_config, Config, "configs.zip")
        simulation.run()
               
class TestEnsemble:
    pass
