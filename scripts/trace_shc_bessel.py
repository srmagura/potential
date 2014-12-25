# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import problems.shc_bessel as shc_bessel

problem = shc_bessel.ShcBesselKnown()
