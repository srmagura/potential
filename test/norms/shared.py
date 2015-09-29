import unittest
import numpy as np

def get_random_v(n_nodes):
    """
    Return a randomly-generated complex-valued function defined on
    `nodes`.
    """
    v_real = np.random.uniform(-1, 1, n_nodes)
    v_imag = np.random.uniform(-1, 1, n_nodes)
    return v_real + 1j*v_imag


class NormTest(unittest.TestCase):

    def _test_spd(self, ip_array):
        """
        Verify that `ip_array` is Hermitian and positive definite.
        """
        ip_matrix = np.matrix(ip_array)

        self.assertTrue(np.array_equal(ip_matrix.getH(), ip_matrix))

        for i in range(25):
            v = get_random_v(ip_matrix.shape[0])
            if np.array_equal(v, np.zeros(len(v))):
                continue

            norm = np.sqrt(np.vdot(v, ip_array.dot(v)))
            self.assertTrue(norm > 0)

    def _test_norm(self, nodes, eval_norm, ip_array):
        """
        Verify that two methods of evaluating the norm yield the same result.

        nodes -- set of nodes on which norm should be computed
        eval_norm -- function that takes a grid function on `nodes` as input
            and returns the norm
        ip_array -- inner product array for the norm
        """
        # TODO Refactor?
        for i in range(10):
            v = get_random_v(len(nodes))
            norm1 = eval_norm(v)
            norm2 = np.vdot(v, ip_array.dot(v))
            self.assertTrue(abs(norm1 - norm2) < 1e-12)
