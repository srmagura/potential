"""
Testing for the norms.l2 module.
"""
import numpy as np

import norms.linalg
import norms.l2
import norms.weight_func

from test.norms.shared import NormTest

class l2TestMixin:

    def test_spd(self):
        self._test_spd(self.ip_array)

    def test_average(self):
        average = np.sum(self.ip_array) / self.ip_array.shape[0]
        self.assertTrue(np.allclose(average, self.h))


class Test_l2_Unweighted(NormTest, l2TestMixin):

    def setUp(self):
        self.h = 0.1
        n_nodes = 4
        self.ip_array = norms.l2.get_ip_array__unweighted(self.h, n_nodes)

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


class Test_l2_Radial(NormTest, l2TestMixin):

    def setUp(self):
        self.h = 0.15
        radii = np.arange(0.1, 1.2, 0.1)
        self.ip_array = norms.l2.get_ip_array__radial(self.h, radii,
            norms.weight_func.wf1)
