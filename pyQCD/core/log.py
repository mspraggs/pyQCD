
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
            break
    return logging.getLogger(inspect.getmodule(frame[0]).__name__)
