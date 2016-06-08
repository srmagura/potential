import numpy as np
from numpy import cos, sin

from domain_util import cart_to_polar

class Problem:

    homogeneous = False
    expected_known = False

    R = 2.3
    AD_len = 2*np.pi

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval_expected(self, x, y, **kwargs):
        r, th = cart_to_polar(x, y)

        if th < self.a/2:
            th += 2*np.pi

        return self.eval_expected_polar(r, th, **kwargs)

    def eval_expected_polar(self, r, th, **kwargs):
        x = r*np.cos(th)
        y = r*np.sin(th)

        return self.eval_expected(x, y, **kwargs)

    def eval_f(self, x, y):
        if self.homogeneous:
            return 0

        return self.eval_f_polar(*cart_to_polar(x, y))

    def eval_f_polar(self, r, th):
        if self.homogeneous:
            return 0

        x = r * cos(th)
        y = r * sin(th)
        return self.eval_f(x, y)


_a = np.pi / 6
_nu = np.pi / (2*np.pi - _a)

class PizzaProblem(Problem):

    a = _a
    nu = _nu

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_n_basis(self, N, **kwargs):
        if hasattr(self, 'n_basis_dict'):
            n_basis_dict = self.n_basis_dict
        else:
            n_basis_dict = self.get_n_basis_dict(**kwargs)

        # If N is not in n_basis_dict, go with the largest N that
        # is in n_basis_dict
        while True:
            if N in n_basis_dict:
                return n_basis_dict[N]
            else:
                N //= 2

    def get_n_basis_dict(self, **kwargs):
        """
        Override this function to make the number of basis functions
        dependent on a parameter other than N (e.g. the order of the
        scheme.)
        """

        # These values probably need to be adjusted for your specific problem
        return {
            16: (21, 9),
            32: (28, 9),
            64: (34, 17),
            128: (40, 24),
            256: (45, 29),
            None: (53, 34)
        }

    def get_m1(self):
        if hasattr(self, 'm1'):
            return self.m1

        bname = self.boundary.name
        if hasattr(self, 'm1_dict') and bname in self.m1_dict:
            return self.m1_dict[bname]

        print('m1 not set')
        return 0
