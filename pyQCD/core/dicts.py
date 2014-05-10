from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

update_methods = {'heatbath': 0,
                  'staple_metropolis': 1,
                  'metropolis': 2}

gauge_actions = {'wilson': 0,
                 'rectangle_improved': 1,
                 'twisted_rectangle_improved': 2}

fermion_actions = {'wilson': 0,
                   'hamber-wu': 1,
                   'naik': 2}

solver_methods = {'bicgstab': 0,
                  'conjugate_gradient': 1,
                  'gmres': 2}
    
smearing_types = {'jacobi': 0}

truefalse = {True: 1,
             False: 0}
