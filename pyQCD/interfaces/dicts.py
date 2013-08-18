
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
