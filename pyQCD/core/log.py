
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import logging
import inspect

def logger():
    """Generates a Logger object with path equal to the module, class and
    function in which this function is called.

    Returns:
      logging.Logger: The logger object.

    Examples:
      Here we create a test function and call logger in it, just to try it
      out.

      >>> import pyQCD
      >>> import logging
      >>> def test(x):
      ...     log = pyQCD.logger()
      ...     log.info("Argument x = {}".format(x))
      ...
      >>> logging.basicConfig(level=logging.INFO)
      >>> test(1):
      >>> INFO:test:Argument x = 1
    """
    
    stack = inspect.stack()
    for frame in stack[1:]:
        if not frame[3].startswith("_"):
            obj = frame[0], frame[3]
            break

    try:
        names = [inspect.getmodule(obj[0]).__name__]
    except AttributeError:
        names = []
    
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

class _Log(object):

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

def Log(func_message=None, ignore=()):
    if callable(func_message):
        log = _Log()
        return log(func_message)
    else:
        def _wrapper(f):
            log = _Log(func_message, ignore)
            return log(f)
        return _wrapper
        
class _ApplyLog(_Log):

    def __init__(self, operator_label):
        self.init_message = "Applying {} operator...".format(operator_label)
        self.ignore = ("self", "psi")

def ApplyLog(func_message=None):
    if callable(func_message):
        log = _ApplyLog()
        return log(func_message)
    else:
        def _wrapper(f):
            log = _ApplyLog(func_message)
            return log(f)
        return _wrapper

class _InversionLog(_Log):

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

def InversionLog(func_message=None):
    if callable(func_message):
        log = _InversionLog()
        return log(func_message)
    else:
        def _wrapper(f):
            log = _InversionLog(func_message)
            return log(f)
        return _wrapper
