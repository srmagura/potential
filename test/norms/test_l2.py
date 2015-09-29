"""
Testing for the norms.l2 module.
"""
import unittest
import numpy as np

import norms.linalg
import norms.l2

import test.norms.shared as shared


class Test_l2_Unweighted(unittest.TestCase, shared.Shared):

    def setUp(self):
        h = 0.1
        nodes = ((-1, 0), (0, 0), (1, 0), (0, 1))
        self.ip_array = norms.l2.get_ip_array__unweighted(h, nodes)

    def test_spd(self):
        self._test_spd(self.ip_array)

    def test_lstsq(self):
        """
        Verify that solve_var() with unweighted l2 yields the same result
        as np.linalg.lstsq().
        """
        A = np.array([
            [1, 2, 3],
            [5, 4, 5],
            [-1, 0, 0],
            [-.5, 3, 1.2],
        ])
        b = np.array([2, 3, 4, -0.5])

        x_solve_var = norms.linalg.solve_var(A, b, self.ip_array)

        numpy_result = np.linalg.lstsq(A, b)
        x_numpy = numpy_result[0]
        residual = numpy_result[1]

        self.assertTrue(np.allclose(x_solve_var, x_numpy))
        self.assertTrue(abs(residual) > 1)
