
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import logging
import inspect

def _logger():
    
    stack = inspect.stack()
    for frame in stack:
        if not frame[3].startswith("_"):
            obj = frame[0], frame[3]
            break
    
    names = [inspect.getmodule(obj[0]).__name__]
    
    try:
        names.append(obj[0].f_locals['self'].__class__.__name__)
    except KeyError:
        pass

    names.append(obj[1])
    
    return logging.getLogger(".".join(names))

def make_func_logger(f, args):

    # First construct a logger based on module, class and function names
    names = [inspect.getmodule(f).__name__]
    if (inspect.isclass(args[0].__class__)
        and "self" in inspect.getargspec(f).args):
        names.append(args[0].__class__.__name__)
    names.append(f.__name__)
    return logging.getLogger(".".join(names))

def merge_arguments(argspec, args, kwargs):

    if argspec.defaults == None:
        return kwargs
    else:        
        kwargs_defaults = dict(zip(argspec.args[-len(argspec.defaults):],
                                   argspec.defaults))
        kwargs_defaults.update(kwargs)

        for key, val in zip(argspec.args, args):
            try:
                kwargs_defaults.pop(key)
            except KeyError:
                pass

        return kwargs_defaults

class Log(object):

    def __init__(self, message=None, ignore=()):
        self.init_message = message
        self.ignore=ignore + ("self",)

    def __call__(self, f):

        def _wrapper(*args, **kwargs):

            logger = make_func_logger(f, args)

            if self.init_message != None:
                logger.info(self.init_message)
            
            argspec = inspect.getargspec(f)
            # List the arguments that don't have defaults
            self.print_args(zip(argspec.args, args), logger)

            # Now merge any default values with the kwargs and list these
            kwargs = merge_arguments(argspec, args, kwargs)
            self.print_args(kwargs.items(), logger)

            return f(*args, **kwargs)

        return _wrapper

    def print_args(self, args, logger):

        for key, val in args:
            if key in self.ignore:
                continue
            logger.info("{}: {}".format(key, val))
        
class ApplyLog(Log):

    def __init__(self, operator_label):
        self.init_message = "Applying {} operator...".format(operator_label)
        self.ignore = ("self", "psi")

class InversionLog(Log):

    def __init__(self, action_label):
        self.init_message = ("Inverting {} Dirac operator..."
                             .format(action_label))
        self.ignore = ("self", "eta", "solver_info")

    def __call__(self, f):

        def _wrapper(*args, **kwargs):

            logger = make_func_logger(f, args)

            if self.init_message != None:
                logger.info(self.init_message)
                
            argspec = inspect.getargspec(f)
            # List the arguments that don't have defaults
            self.print_args(zip(argspec.args, args), logger)

            # Now merge any default values with the kwargs and list these
            kwargs = merge_arguments(argspec, args, kwargs)
            kwargs.update({"solver_info": True})
            self.print_args(kwargs.items(), logger)

            result = f(*args, **kwargs)

            logger.info("Solver finished after {} iterations with residual {}"
                        .format(result[1], result[2]))
            logger.info("CPU time used: {}".format(result[3]))

            return result[0]

        return _wrapper
