import numpy as np

import domain_util

from .problem import PizzaProblem

class SmoothProblem(PizzaProblem):

    expected_known = True
    regularize = False

    k = 1

    n_basis_dict = {
        16: (24, 6),
        32: (33, 8),
        64: (42, 12),
        128: (65, 18),
        256: (80, 30),
        512: (100, 40),
        1024: (120, 45),
    }

    def eval_bc(self, arg, sid):
        """
        Evaluate extended boundary condition.
        """
        r, th = domain_util.arg_to_polar(self.R, self.a, arg, sid)
        return self.eval_expected_polar(r, th)


class Smooth_H(SmoothProblem):

    homogeneous = True

    def eval_expected(self, x, y):
        k = self.k

        kx = .8*k
        ky = .6*k

        return np.sin(k*x)
        #return np.exp(1j*(kx*x + ky*y))

    def eval_d_u_outwards(self, arg, sid):
        k = self.k
        a = self.a

        r, th = domain_util.arg_to_polar(self.R, a, arg, sid)
        x, y = r*np.cos(th), r*np.sin(th)

        grad = (k*np.cos(k*x), 0)

        if sid == 0:
            normal = (np.cos(th), np.sin(th))
        elif sid == 1:
            normal = (0, 1)
        elif sid == 2:
            normal = (np.sin(a), -np.cos(a))

        return np.vdot(grad, normal)
