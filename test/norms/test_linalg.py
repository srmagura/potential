"""
Testing for the norms.linalg module.
"""
import unittest
import numpy as np
import scipy.optimize

import norms.sobolev
from norms.linalg import solve_var, eval_norm

class TestLinalg(unittest.TestCase):

    def _test_minimize(self, A, b, nodes, tol):
        """
        Compare the results of solve_var() to
        scipy.optimize.minimize().

        This test passes the ip_array for the Sobolev norm to solve_var().

        A -- left-hand side of the overdetermined linear system
        b -- right-hand side of the linear system
        nodes -- set of nodes, required for evaluating the Sobolev norm
        tol -- require the two solutions to be within this tolerance
            of each other for the test to pass. Difference between
            solutions is computed in the infinty norm.

        Not sure if this function will still work if A, b, or the weak
        solution to Ax=b contain non-real numbers.
        """
        h = 0.1
        sa = 0.5

        ip_array = norms.sobolev.get_ip_array(h, nodes, sa)
        x1 = solve_var(A, b, ip_array)

        x1_res = A.dot(x1) - b
        x1_res_norm = np.vdot(x1_res, ip_array.dot(x1_res))

        def to_minimize(x):
            residual = A.dot(x) - b
            return eval_norm(ip_array, residual)

        guess = np.zeros(A.shape[1])
        opt_result = scipy.optimize.minimize(to_minimize, guess)
        x2 = opt_result.x
        diff = np.max(np.abs(x1-x2))

        #print('Residual norms:', x1_res_norm, to_minimize(x2))
        #print('Difference:', diff)

        self.assertTrue(diff < tol)

    def test_minimize1(self):
        nodes = ((0, 0),)
        A = np.array([[2]])
        b = np.array([1])
        self._test_minimize(A, b, nodes, 1e-8)

    def test_minimize3(self):
        nodes = ((-1, 0), (0, 0), (1, 0))
        A = np.array([
            [1, 2],
            [0, 1],
            [1.05, 1.99]
        ])
        b = np.array([-1, 1/2, -1])
        self._test_minimize(A, b, nodes, 1e-6)

    def test_minimize4(self):
        nodes = ((-1, 0), (0, 0), (1, 0), (0, 1))
        A = np.array([
            [0.5, 2, -1],
            [3, 4, 5],
            [-2.5, 1.1, 2.24],
            [0.51, 2.05, -1.04],
        ])
        b = np.array([1, -3, 5, 1])
        self._test_minimize(A, b, nodes, 1e-6)
