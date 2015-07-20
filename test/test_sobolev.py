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

    def setUp(self):
        self.h = 0.1
        self.sa = 0.5
        self.nodes = ((-1, 0), (0, 0), (1, 0), (0, 1))

    def get_random_v(self):
        """
        Return a randomly-generated complex-valued function defined on
        self.nodes.
        """
        v_real = np.random.rand(len(self.nodes))
        v_imag = np.random.rand(len(self.nodes))
        return v_real + 1j*v_imag

    def get_random_A(self, dim2):
        """
        Return a randomly-generated linear operator (matrix) that acts
        on grid functions defined on self.nodes.

        The dimensions of the matrix are (len(self.nodes), dim2).
        """
        columns = []
        for i in range(dim2):
            columns.append(self.get_random_v())

        return np.column_stack(columns)

    def test_spd(self):
        """
        Verify that ip_array is Hermitian and positive definite.
        """
        ip_array = sobolev.get_ip_array(self.h, self.nodes, self.sa)
        ip_matrix = np.matrix(ip_array)

        self.assertTrue(np.array_equal(ip_matrix.getH(), ip_matrix))

        for i in range(25):
            v = self.get_random_v()
            if np.array_equal(v, np.zeros(len(v))):
                continue

            norm = np.vdot(v, ip_array.dot(v))
            self.assertTrue(norm > 0)

    def test_norm(self):
        """
        Test that the two methods of evaluating the Sobolev norm
        eval_norm() and sobolev.get_ip_array() return the same results.
        """
        ip_array = sobolev.get_ip_array(self.h, self.nodes, self.sa)

        for i in range(10):
            v = self.get_random_v()
            norm1 = eval_norm(self.h, self.nodes, self.sa, v)
            norm2 = np.vdot(v, ip_array.dot(v))
            self.assertTrue(abs(norm1 - norm2) < 1e-13)

    def test_minimize(self):
        """
        Compare the results of sobolev.solve_var() to
        scipy.optimize.minimize().
        """
        dim2 = 4
        A = self.get_random_A(dim2)
        b = self.get_random_v()

        x1 = sobolev.solve_var(A, b, self.h, self.nodes, self.sa)

        def to_minimize(x):
            residual = A.dot(x) - b
            return eval_norm(self.h, self.nodes, self.sa, residual)

        guess = np.zeros(dim2)
        opt_result = scipy.optimize.minimize(to_minimize, guess)
        print(opt_result)
        x2 = opt_result.x

        print(abs(x1-x2))
