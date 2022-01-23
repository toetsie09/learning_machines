# from base import Robobo
from __future__ import absolute_import, print_function
from .simulation import SimulationRobobo
from .simulation_prey import SimulationRoboboPrey

from .hardware import HardwareRobobo

# See:
# https://canvas.vu.nl/courses/56526/discussion_topics/479043

#import sys
#if sys.version_info < (3,0):
#    from .hardware import HardwareRobobo
#else:
#    print("Hardware Connection not available in python3 :(", file=sys.stderr)