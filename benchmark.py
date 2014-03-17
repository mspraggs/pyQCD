import time
import sys
import inspect

import numpy as np

import pyQCD

def time_function(function, min_trials=10, args=[]):
    
    t0 = time.time()
    function(*args)
    tf = time.time()
    
    n_trials = int(1.0 / (tf - t0))
    n_trials = min_trials if n_trials < min_trials else n_trials
    
    times = np.zeros(n_trials)
    
    for i in xrange(n_trials):
        t0 = time.time()
        function(*args)
        tf = time.time()
        times[i] = tf - t0
        
    return times.mean(), times.std() / times.mean() * 100, n_trials

def display_benchmark(results=None, function=None, min_trials=10, args=[]):
    
    num_spaces = 40 - len("Function {}:".format(function.__name__))
    arg_names = function.func_code.co_varnames
    arg_names \
      = arg_names[1:len(args)+1] \
      if arg_names[0] == "self" else arg_names[len(args)]
        
    print("Function: {}".format(function.__name__))
    if len(args) > 0:
        print("Arguments:")
        for name_val_pair in zip(arg_names, [x.__repr__() for x in args]):
            print("  {} = {}".format(*name_val_pair))
    
    if results != None:
        t_bar, t_sigma, n_trials = results
    elif function != None:
        t_bar, t_sigma, n_trials = time_function(function, args=args)
            
    print("Time: {:f} s +/- {:f} % ({} trials)"
          .format(t_bar, t_sigma, n_trials))
    
    return t_bar, t_sigma, n_trials

def benchmark_gauge_actions(L=8, T=8):
    
    print("Benchmarking Gauge Actions")
    print("--------------------------")
    print("Lattice shape = ({0}, {1}, {1}, {1})".format(T, L))
    print("")
    
    actions = ["wilson", "rectangle_improved"]
    
    for action in actions:
        lattice = pyQCD.Lattice(L=L, T=T, action=action)
        print("Action: {}".format(action))
        display_benchmark(function=lattice.update)
        print("")

def benchmark_update_algorithms(L=8, T=8):
    
    print("Benchmarking Update Algorithms")
    print("------------------------------")
    print("Lattice shape = ({0}, {1}, {1}, {1})".format(T, L))
    print("")
    
    algorithms = ["staple_metropolis", "heatbath"]
    
    for algorithm in algorithms:
        lattice = pyQCD.Lattice(L=L, T=T, update_method=algorithm)
            
        print("Algorithm: {}".format(algorithm))
        
        av_plaquette = 1.0
        t_total = 0
        while av_plaquette > 0.5:
            t0 = time.time()
            lattice.update()
            tf = time.time()
            t_total += tf - t0
            av_plaquette = lattice.get_av_plaquette()
            
        print("Time to thermalize: {} s".format(t_total))
        print("")

def benchmark_parallel_update(L=8, T=8):
    
    print("Benchmarking Parallel Update")
    print("----------------------------")
    print("Lattice shape = ({0}, {1}, {1}, {1})".format(T, L))
    print("")
    
    lattice = pyQCD.Lattice(L=L, T=T, parallel_updates=False)
    print("Not using OpenMP")
    single_results = display_benchmark(function=lattice.update)
    print("")
    
    lattice = pyQCD.Lattice(L=L, T=T, parallel_updates=True)
    print("Using OpenMP")
    parallel_results = display_benchmark(function=lattice.update)
    print("")
    
    print("Speed-up: {:f} x".format(single_results[0] / parallel_results[0]))
    print("")
    
def benchmark_wilson_loops(L=8, T=8):
    
    print("Benchmarking Wilson Loop Computation")
    print("------------------------------------")
    print("Lattice shape = ({0}, {1}, {1}, {1})".format(T, L))
    print("")
    
    lattice = pyQCD.Lattice(L=L, T=T, parallel_updates=False)
    print("Not using OpenMP")
    single_results = display_benchmark(function=lattice.get_wilson_loops)
    print("")
    
    lattice = pyQCD.Lattice(L=L, T=T, parallel_updates=True)
    print("Using OpenMP")
    parallel_results = display_benchmark(function=lattice.get_wilson_loops)
    print("")
    
    print("Speed-up: {:f} x".format(single_results[0] / parallel_results[0]))
    print("")
    
    for n in xrange(3):
        display_benchmark(function=lattice.get_wilson_loops, args=[n, 0.4])
        print("")
    
def benchmark_propagator_computation(L=4, T=8):
    
    print("Benchmarking Propagator Computation")
    print("-----------------------------------")
    print("Lattice shape = ({0}, {1}, {1}, {1})".format(T, L))
    print("")

    lattice = pyQCD.Lattice(L=L, T=T)
    lattice.update()
    
    display_benchmark(function=lattice.get_wilson_propagator,
                      args=[0.4, [0, 0, 0, 0], 0, 1.0, "jacobi", 0, 1.0,
                            "jacobi", 0, 1.0, "conjugate_gradient"])
    print("")
    
    display_benchmark(function=lattice.get_wilson_propagator,
                      args=[0.4, [0, 0, 0, 0], 0, 1.0, "jacobi", 0, 1.0,
                            "jacobi", 0, 1.0, "bicgstab"])
    print("")
    
    for n in xrange(1, 3):
        display_benchmark(function=lattice.get_wilson_propagator,
                          args=[0.4, [0, 0, 0, 0], n, 1.0, "jacobi", 0, 1.0,
                                "jacobi", 0, 1.0, "bicgstab"])
        print("")
    
    for n in xrange(1, 3):
        display_benchmark(function=lattice.get_wilson_propagator,
                          args=[0.4, [0, 0, 0, 0], 0, 1.0, "jacobi", n, 1.0,
                                "jacobi", 0, 1.0, "bicgstab"])
        print("")
    
    for n in xrange(1, 3):
        display_benchmark(function=lattice.get_wilson_propagator,
                          args=[0.4, [0, 0, 0, 0], 0, 1.0, "jacobi", 0, 1.0,
                                "jacobi", n, 1.0, "bicgstab"])
        print("")
        
def benchmark_twopoint_computation():
    
    print("Benchmarking Two-point Computation")
    print("----------------------------------")
    
    prop_data \
      = np.load("pyQCD/test/data/propagator_purgaug_4c8_4000_no_smear.npy")
      
    prop = pyQCD.Propagator(prop_data, 4, 8, 5.5, 1.0, "wilson", 0.4,
                            [0, 0, 0, 0], 0, 1.0, 0, 1.0, 0, 1.0)
    
    twopoint = pyQCD.TwoPoint(8, 4)
    
    display_benchmark(function=twopoint.compute_meson_correlator,
                      args=[prop, prop, "g5", "g5", "pion"])
    print("")
    
    display_benchmark(function=twopoint.compute_all_meson_correlators,
                      args=[prop, prop])
    print("")
        
def benchmark_twopoint_fit():
    
    print("Benchmarking Two-point Fitting")
    print("------------------------------")
    
    data = pyQCD.DataSet.load("pyQCD/test/data/Wilson_pt_src_folded.zip")
    
    twopoint = data[0]
    display_benchmark(function=twopoint.compute_energy,
                      args=[[10, 16], [1.0, 1.0], None, "PS_PS",
                            [0.3, 0.3], [0, 0, 0]])
    print("")
    
def benchmark_dataset_jackknife():
    
    print("Benchmarking Jackknife Fitting")
    print("------------------------------")
    
    data = pyQCD.DataSet.load("pyQCD/test/data/Wilson_pt_src_folded.zip")
    data.num_data = 20
    
    print("Dataset size = {}".format(data.num_data))
    print("")
    
    for binsize in [1, 2, 5]:
    
        display_benchmark(function=data.generate_jackknife_cache,
                          args=[binsize])
        print("")
    
        display_benchmark(function=data.jackknife,
                          args=[pyQCD.TwoPoint.compute_energy, binsize,
                                [[10, 16], [1.0, 1.0], None, "PS_PS",
                                 [0.3, 0.3], [0, 0, 0]],
                                 False])
        print("")
    
        display_benchmark(function=data.jackknife,
                          args=[pyQCD.TwoPoint.compute_energy, binsize,
                                [[10, 16], [1.0, 1.0], None, "PS_PS",
                                 [0.3, 0.3], [0, 0, 0]],
                                 True])
        print("")

def benchmark_dataset_bootstrap():
    
    print("Benchmarking Bootstrap Fitting")
    print("------------------------------")
    
    data = pyQCD.DataSet.load("pyQCD/test/data/Wilson_pt_src_folded.zip")
    data.num_data = 20
    
    print("Dataset size = {}".format(data.num_data))
    print("")
    
    for binsize in [1, 2, 5]:
    
        display_benchmark(function=data.generate_bootstrap_cache,
                          args=[20, binsize])
        print("")
    
        display_benchmark(function=data.bootstrap,
                          args=[pyQCD.TwoPoint.compute_energy, 20, binsize,
                                [[10, 16], [1.0, 1.0], None, "PS_PS",
                                 [0.3, 0.3], [0, 0, 0]],
                                 False])
        print("")
    
        display_benchmark(function=data.bootstrap,
                          args=[pyQCD.TwoPoint.compute_energy, 20, binsize,
                                [[10, 16], [1.0, 1.0], None, "PS_PS",
                                 [0.3, 0.3], [0, 0, 0]],
                                 True])
        print("")
    
if __name__ == "__main__":
    
    try:
        filter_string = sys.argv[1]
    except IndexError:
        filter_string = "benchmark"
    
    functions \
      = [obj for name, obj in inspect.getmembers(sys.modules[__name__])
         if name.find(filter_string) > -1]
             
    for function in functions:
        try:
            function()
        except AttributeError:
            print("Skipping function {}.\n(Have the simulation components been "
                  "built?)".format(function.__name__))
