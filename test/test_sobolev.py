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
    return h*norm


class TestSobolev(unittest.TestCase):

    h = 0.1
    sa = 0.5

    def setUp(self):
        pass

    def get_random_v(self, nodes):
        """
        Return a randomly-generated complex-valued function defined on
        self.nodes.
        """
        v_real = np.random.rand(len(nodes))
        v_imag = np.random.rand(len(nodes))
        return v_real + 1j*v_imag

    def test_spd(self):
        """
        Verify that ip_array is Hermitian and positive definite.
        """
        nodes = ((-1, 0), (0, 0), (1, 0), (0, 1))
        ip_array = sobolev.get_ip_array(self.h, nodes, self.sa)
        ip_matrix = np.matrix(ip_array)

        self.assertTrue(np.array_equal(ip_matrix.getH(), ip_matrix))

        for i in range(25):
            v = self.get_random_v(nodes)
            if np.array_equal(v, np.zeros(len(v))):
                continue

            norm = np.vdot(v, ip_array.dot(v))
            self.assertTrue(norm > 0)

    def test_norm(self):
        """
        Test that the two methods of evaluating the Sobolev norm
        eval_norm() and sobolev.get_ip_array() return the same results.
        """
        nodes = ((-1, 0), (0, 0), (1, 0), (0, 1))
        ip_array = sobolev.get_ip_array(self.h, nodes, self.sa)

        for i in range(10):
            v = self.get_random_v(nodes)
            norm1 = eval_norm(self.h, nodes, self.sa, v)
            norm2 = np.vdot(v, ip_array.dot(v))
            self.assertTrue(abs(norm1 - norm2) < 1e-13)

    def _test_minimize(self, A, b, nodes, tol):
        """
        Compare the results of sobolev.solve_var() to
        scipy.optimize.minimize().

        A -- left-hand side of the overdetermined linear system
        b -- right-hand side of the linear system
        nodes -- set of nodes, required for evaluating the Sobolev norm
        tol -- require the two solutions to be within this tolerance
            of each other for the test to pass. Difference between
            solutions is computed in the infinty norm.

        Not sure if this function will still work if A, b, or the weak
        solution to Ax=b contains non-real numbers.
        """
        x1 = sobolev.solve_var(A, b, self.h, nodes, self.sa)

        ip_array = sobolev.get_ip_array(self.h, nodes, self.sa)
        x1_res = A.dot(x1) - b
        x1_res_norm = np.vdot(x1_res, ip_array.dot(x1_res))

        def to_minimize(x):
            residual = A.dot(x) - b
            return eval_norm(self.h, nodes, self.sa, residual)

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
        self._test_minimize(A, b, nodes, 1e-7)

    def test_minimize4(self):
        nodes = ((-1, 0), (0, 0), (1, 0), (0, 1))
        A = np.array([
            [0.5, 2, -1],
            [3, 4, 5],
            [-2.5, 1.1, 2.24],
            [0.51, 2.05, -1.04],
        ])
        b = np.array([1, -3, 5, 1])
        self._test_minimize(A, b, nodes, 1e-3)
