
update_methods = {'HEATBATH': 0,
                  'STAPLE_MONTE_CARLO': 1,
                  'MONTE_CARLO': 2}

gauge_actions = {'WILSON': 0,
                 'RECTANGLE_IMPROVED': 1,
                 'TWISTED_RECTANGLE_IMPROVED': 2}

solver_methods = {'BiCGSTAB': 0,
                  'ConjugateGradient': 1}

truefalse = {'TRUE': 1,
             'FALSE': 0}

data_types = {'PLAQUETTE': 0,
              'WILSON_LOOP': 1,
              'PROPAGATOR': 2}

colours = {'BLACK': 'k',
           'BLUE': 'b',
           'RED': 'r',
           'GREEN': 'g'}

linestyles = {'SOLID': '-',
              'DASHED': '--',
              'DOTTED': ':',
              'DASHED_DOTTED': '-.'}

dicts = [update_methods,
         gauge_actions,
         solver_methods,
         truefalse,
         data_types,
         colours,
         linestyles]

defaults = {
    'gauge_action': {
        'beta': None,
        'type': None,
        'u0': 1.0
        },
    'lattice': {
        'L': None,
        'T': None
        },
    'measurements': {
        'plaquette': {
            'filename': None
            },
        'propagator': {
            'a': 1.0,
            'field_smearing_param': 1.0,
            'filename': None,
            'mass': None,
            'num_field_smears': 0,
            'num_sink_smears': 0,
            'num_source_smears': 0,
            'sink_smearing_param': 1.0,
            'solver_method': 'BiCGSTAB',
            'source_site': [0, 0, 0, 0],
            'source_smearing_param': 1.0,
            'verbose_output': 'FALSE'
            },
        'wilson_loop': {
            'field_smearing_param': 1.0,
            'filename': None,
            'num_field_smears': 0,
            'r_max': None,
            't_max': None
            },
        'configuration': {
            'filename': None
            }
        },
    'postprocess': {
        'auto_correlation': {
            'bin_size': 0,
            'filename': None,
            'input': [{
                'filename': None,
                'type': None
                }],
            'num_bootstraps': 0
            },
        'lattice_spacing': {
            'bin_size': 0,
            'filename': None,
            'input': [{
                'filename': None,
                'type': None
                }],
            'num_bootstraps': 0
            },
        'pair_potential': {
            'bin_size': 0,
            'filename': None,
            'input': [{
                'filename': None,
                'type': None
                }],
            'num_bootstraps': 0
            }
        },
    'simulation': {
        'ensemble': None,
        'measurement_spacing': None,
        'num_configurations': None,
        'num_warmup_updates': None,
        'parallel_update': {
            'block_size': 1,
            'enabled': 'FALSE'
            },
        'timing_run': {
            'enabled': 'FALSE',
            'num_configurations': 0
        },
        'update_method': 'HEATBATH'
        }
    }
