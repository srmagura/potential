"""
Testing for the sobolev module.
"""
import unittest
import numpy as np
import scipy.optimize

import sobolev

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
    return norm


class TestSobolev(unittest.TestCase):

    h = 0.1
    sa = 0.5
