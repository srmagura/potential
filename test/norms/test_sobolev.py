"""
Testing for the sobolev module.
"""
import numpy as np
import scipy.optimize

import norms.sobolev
from test.norms.shared import NormTest

def eval_norm(h, nodes, sa, v):
    """
    Evaluate the Sobolev norm in the easiest to understand way. Compute
    the three sums over their respective ranges.
    """
    norm = 0

    for l in range(len(nodes)):
        norm += abs(v[l])**2

    deriv_sum = 0
    for k in range(len(nodes)):
        i, j = nodes[k]

        if (i+1, j) in nodes:
            l = nodes.index((i+1, j))
            deriv_sum += abs((v[l] - v[k])/h)**2

        if (i, j+1) in nodes:
            l = nodes.index((i, j+1))
            deriv_sum += abs((v[l] - v[k])/h)**2

    norm += sa * deriv_sum
    return h*norm


class TestSobolev(NormTest):

    h = 0.1
    sa = 0.5

    def test_spd(self):
        nodes = ((-1, 0), (0, 0), (1, 0), (0, 1))
        ip_array = norms.sobolev.get_ip_array(self.h, nodes, self.sa)
        self._test_spd(ip_array)

    def test_norm(self):
        """
        Test that the two methods of evaluating the Sobolev norm,
        eval_norm() and sobolev.get_ip_array(), return the same results.
        """
        nodes = ((-1, 0), (0, 0), (1, 0), (0, 1))

        def _eval_norm(v):
            return eval_norm(self.h, nodes, self.sa, v)

        ip_array = norms.sobolev.get_ip_array(self.h, nodes, self.sa)
        self._test_norm(nodes, _eval_norm, ip_array)
