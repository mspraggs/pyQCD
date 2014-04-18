import pyQCD
import numpy as np

def create_fullpath(fname):
    
    return "pyQCD/test/data/{}".format(fname)

def generate_configs():
    
    # Generate some configs and save the raw data
    for gauge_action in pyQCD.dicts.gauge_actions.keys():
        for update_method in pyQCD.dicts.update_methods.keys():
            for rand_seed in [0, 1, 2]:
                print("Generating config for action {}, "
                      "update method {} and random seed {}"
                      .format(gauge_action, update_method, rand_seed))
                
                filename = "config_{}_{}_{}" \
                  .format(gauge_action,
                          update_method,
                          rand_seed)
                
                lattice = pyQCD.Lattice(rand_seed=rand_seed,
                                        action=gauge_action,
                                        update_method=update_method)
                lattice.update()
                np.save(create_fullpath(filename),
                        lattice.get_config().data)
                
def generate_props():
    
    smearing_combinations \
      = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    
    config_data = np.load(create_fullpath("chroma_config.npy"))
    config = pyQCD.Config(config_data, 4, 8, 5.5, 1.0, 1.0, 1.0, "wilson")
    lattice = pyQCD.Lattice()
    lattice.set_config(config)
    
    function_dict = {"wilson": pyQCD.Lattice.get_wilson_propagator,
                     "hamber-wu": pyQCD.Lattice.get_hamberwu_propagator}
    
    # Generate some props based on one of the configs generated
    for fermion_action in pyQCD.dicts.fermion_actions.keys():
        for solver_method in pyQCD.dicts.solver_methods.keys():
            for smearing_type in pyQCD.dicts.smearing_types.keys():
                for n_link_s, n_source_s, n_sink_s in smearing_combinations:
                    
                    print("Generating propagator for fermion action {}, "
                          "solver method {}, smearing type {}, {} link "
                          "smears, {} source smears and {} sink smears"
                          .format(fermion_action, solver_method, smearing_type,
                                  n_link_s, n_source_s, n_sink_s))
                    
                    func = function_dict[fermion_action]
                    prop = func(lattice, 0.4,
                                num_field_smears=n_link_s,
                                field_smearing_param=0.4,
                                source_smear_type=smearing_type,
                                num_source_smears=n_source_s,
                                source_smearing_param=0.4,
                                sink_smear_type=smearing_type,
                                num_sink_smears=n_sink_s,
                                sink_smearing_param=0.4,
                                solver_method=solver_method,
                                verbosity=2)
                    
                    filename = "prop_{}_{}_{}_{}_{}_{}" \
                      .format(fermion_action, solver_method,
                              smearing_type, n_link_s, n_source_s,
                              n_sink_s)
                    
                    np.save(create_fullpath(filename), prop.data)
                
def generate_spinors():
    
    config_data = np.load(create_fullpath("chroma_config.npy"))
    config = pyQCD.Config(config_data, 4, 8, 5.5, 1.0, 1.0, 1.0, "wilson")
    lattice = pyQCD.Lattice()
    lattice.set_config(config)
    
    function_dict = {"wilson": pyQCD.Lattice.invert_wilson_dirac,
                     "hamber-wu": pyQCD.Lattice.invert_hamberwu_dirac}
        
    psi = np.zeros((8, 4, 4, 4, 4, 3), dtype=np.complex)
    psi[0, 0, 0, 0, 0, 0] = 1.0
    
    psi5d = np.zeros((4, 8, 4, 4, 4, 4, 3), dtype=np.complex)
    psi5d[0] = psi
    
    # Generate some props based on one of the configs generated
    for fermion_action in pyQCD.dicts.fermion_actions.keys():
        for solver_method in pyQCD.dicts.solver_methods.keys():
                
            print("Generating spinor for fermion action {} and "
                  "solver method {}"
                  .format(fermion_action, solver_method))
            
            func = function_dict[fermion_action]
            eta = func(lattice, psi, 0.4,
                       solver_method=solver_method)
            
            filename = "spinor_{}_{}" \
              .format(fermion_action, solver_method)
            
            np.save(create_fullpath(filename), eta)
            
            func = pyQCD.Lattice.invert_dwf_dirac
            eta = func(lattice, psi5d, 0.4, 1.6, 4, fermion_action,
                       solver_method=solver_method)
            
            filename = "spinor_dwf_{}_{}" \
              .format(fermion_action, solver_method)
            
            np.save(create_fullpath(filename), eta)
                
def generate_Dpsis():
    
    config_data = np.load(create_fullpath("chroma_config.npy"))
    config = pyQCD.Config(config_data, 4, 8, 5.5, 1.0, 1.0, 1.0, "wilson")
    lattice = pyQCD.Lattice()
    lattice.set_config(config)
    
    function_dict = {"wilson": pyQCD.Lattice.apply_wilson_dirac,
                     "hamber-wu": pyQCD.Lattice.apply_hamberwu_dirac,
                     "jacobi": pyQCD.Lattice.apply_jacobi_smearing}
        
    psi = np.zeros((8, 4, 4, 4, 4, 3), dtype=np.complex)
    psi[0, 0, 0, 0, 0, 0] = 1.0
    
    psi5d = np.zeros((4, 8, 4, 4, 4, 4, 3), dtype=np.complex)
    psi5d[0] = psi
    
    # Generate some props based on one of the configs generated
    for fermion_action in pyQCD.dicts.fermion_actions.keys():
            
        print("Generating D*psi for fermion action {}"
              .format(fermion_action))
        
        func = function_dict[fermion_action]
        eta = func(lattice, psi, 0.4)
        
        filename = "Dpsi_{}" \
          .format(fermion_action)
        
        np.save(create_fullpath(filename), eta)
        
        func = pyQCD.Lattice.apply_dwf_dirac
        eta = func(lattice, psi5d, 0.4, 1.6, 4, fermion_action)
        
        filename = "Dpsi_dwf_{}" \
          .format(fermion_action)
        
        np.save(create_fullpath(filename), eta)
        
    for smear_type in pyQCD.dicts.smearing_types.keys():
        
        print("Generating smeared source for smearing type {}"
              .format(smear_type))
        
        func = function_dict[smear_type]
        eta = func(lattice, psi, 1, 0.5)
        
        filename = "smeared_source_{}" \
          .format(smear_type)
        
        np.save(create_fullpath(filename), eta)
        
if __name__ == "__main__":
    generate_configs()
    generate_props()
    generate_spinors()
    generate_Dpsis()
