import numpy as np
from scipy.special import jv, jvp

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
        r, th = domain_util.arg_to_polar(self.boundary, self.a, arg, sid)
        return self.eval_expected_polar(r, th)

    def eval_d_u_outwards(self, arg, sid):
        k = self.k
        a = self.a

        r, th = domain_util.arg_to_polar(self.boundary, a, arg, sid)
        x, y = r*np.cos(th), r*np.sin(th)

        grad = self.get_grad(x, y)

        if sid == 0:
            normal = self.boundary.eval_normal(th)
        elif sid == 1:
            normal = (0, 1)
        elif sid == 2:
            normal = (np.sin(a), -np.cos(a))

        return np.dot(grad, normal)



class SmoothH_Sine(SmoothProblem):

    homogeneous = True

    def eval_expected(self, x, y):
        return np.sin(self.k*x)

    def get_grad(self, x, y):
        k = self.k
        return (k*np.cos(k*x), 0)


class SmoothH_E(SmoothProblem):

    k = 1.75
    homogeneous = True

    def eval_expected(self, x, y):
        k = self.k

        self.kx = kx = .8*k
        self.ky = ky = .6*k

        return np.exp(1j*(kx*x + ky*y))

    def get_grad(self, x, y):
        kx = self.kx
        ky = self.ky

        u = np.exp(1j*(kx*x + ky*y))
        return (1j*kx*u, 1j*ky*u)
