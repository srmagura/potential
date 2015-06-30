import numpy as np
from numpy import cos, sin

from solver import cart_to_polar
from ps.ps import PizzaSolver

class Problem:
    homogeneous = False
    var_compute_b = False
    expected_known = False
    force_relative = False

    R = 2.3
    AD_len = 2*np.pi

    def __init__(self, **kwargs):
        super().__init__()

    def get_solver(self, *args, **kwargs):
        return self.solver_class(self, *args, **kwargs)

    def eval_expected(self, x, y, **kwargs):
        r, th = cart_to_polar(x, y)

        if th < self.a/2:
            th += 2*np.pi

        return self.eval_expected_polar(r, th, **kwargs)

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
    solver_class = PizzaSolver

    # These values probably need to be adjusted for your specific problem
    n_basis_dict = {
        16: (21, 9),
        32: (28, 9),
        64: (34, 17),
        128: (40, 24),
        256: (45, 29),
        None: (53, 34)
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def wrap_func(self, arg, sid):
        '''
        Can be used to create the smooth extensions of the Dirichlet data
        needed for the Chebyshev fit, for certain problems. In particular,
        the Dirichlet data needs to be sufficiently smooth at the reentrant
        corner.
        '''
        if sid == 0 and arg < self.a/2:
                arg += 2*np.pi

        elif sid == 1 and arg < 0:
            arg = -arg
            sid = 2

        elif sid == 2 and arg < 0:
            arg = -arg
            sid = 1

        return (arg, sid)

    def arg_to_polar(self, arg, sid):
        if sid == 0:
            return (self.R, arg)
        elif sid == 1:
            if arg >= 0:
                return (arg, 2*np.pi)
            else:
                return (-arg, np.pi)
        elif sid == 2:
            if arg >= 0:
                return (arg, self.a)
            else:
                return (-arg, np.pi + self.a)

    def get_sid(self, th):
        '''
        Mapping of polar angles in the interval (0, 2*pi] to the set of
        segment ID's, {0, 1, 2}.
        '''
        tol = 1e-12
        a = self.a

        if th >= 2*np.pi:
            return 1
        elif th > a:
            return 0
        elif abs(th - a) < tol:
            return 2

    def get_n_basis(self, N):
        if N in self.n_basis_dict:
            return self.n_basis_dict[N]
        else:
            return self.n_basis_dict[None]

    def get_restore_polar(self, r, th):
        return 0
