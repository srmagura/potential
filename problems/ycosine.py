import numpy as np
from numpy import cos, sin

from solver import cart_to_polar

from .problem import PizzaProblem


class YCosine(PizzaProblem):

    # Note: this problem becomes homogeneous if k=1.
    k = 1.75

    expected_known = True

    n_basis_dict = {
        16: (23, 7),
        32: (25, 7),
        64: (31, 15),
        128: (37, 21),
        None: (45, 25),
    }

    def eval_expected(self, x, y):
        return y*cos(x)

    def eval_bc_extended(self, arg, sid):
        r, th = self.arg_to_polar(arg, sid)
        x, y = r*cos(th), r*sin(th)

        return self.eval_expected(x, y)

    def eval_d_u_outwards(self, arg, sid):
        a = self.a

        r, th = self.arg_to_polar(arg, sid)
        x, y = r*cos(th), r*sin(th)

        if sid == 0:
            return self.eval_d_u_r(th)

        elif sid == 1:
            return cos(x)

        elif sid == 2:
            d_u_x = -y*sin(x)
            d_u_y = cos(x)

            b = np.pi/2 - a
            return d_u_x*cos(b)-d_u_y*sin(b)

    def eval_f(self, x, y):
        k = self.k
        return k**2*y*cos(x) - y*cos(x)

    def eval_d_f_r(self, r, th):
        k = self.k
        return -k**2*r*sin(th)*sin(r*cos(th))*cos(th) + k**2*sin(th)*cos(r*cos(th)) + r*sin(th)*sin(r*cos(th))*cos(th) - sin(th)*cos(r*cos(th))

    def eval_d2_f_r(self, r, th):
        k = self.k
        return (-k**2*r*cos(th)*cos(r*cos(th)) - 2*k**2*sin(r*cos(th)) + r*cos(th)*cos(r*cos(th)) + 2*sin(r*cos(th)))*sin(th)*cos(th)

    def eval_d_f_th(self, r, th):
        k = self.k
        return k**2*r**2*sin(th)**2*sin(r*cos(th)) + k**2*r*cos(th)*cos(r*cos(th)) - r**2*sin(th)**2*sin(r*cos(th)) - r*cos(th)*cos(r*cos(th))

    def eval_d2_f_th(self, r, th):
        k = self.k
        return r*(-k**2*r**2*sin(th)**2*cos(r*cos(th)) + 3*k**2*r*sin(r*cos(th))*cos(th) - k**2*cos(r*cos(th)) + r**2*sin(th)**2*cos(r*cos(th)) - 3*r*sin(r*cos(th))*cos(th) + cos(r*cos(th)))*sin(th)

    def eval_d2_f_r_th(self, r, th):
        k = self.k
        return k**2*r**2*sin(th)**2*cos(th)*cos(r*cos(th)) + 2*k**2*r*sin(th)**2*sin(r*cos(th)) - k**2*r*sin(r*cos(th))*cos(th)**2 + k**2*cos(th)*cos(r*cos(th)) - r**2*sin(th)**2*cos(th)*cos(r*cos(th)) - 2*r*sin(th)**2*sin(r*cos(th)) + r*sin(r*cos(th))*cos(th)**2 - cos(th)*cos(r*cos(th))

    def eval_grad_f(self, x, y):
        k = self.k
        return np.array((-k**2*y*sin(x) + y*sin(x), k**2*cos(x) - cos(x)))

    def eval_hessian_f(self, x, y):
        k = self.k
        return np.array(((y*(-k**2 + 1)*cos(x), (-k**2 + 1)*sin(x)), ((-k**2 + 1)*sin(x), 0)))

    def eval_d_u_r(self, th):
        R = self.R
        x = R*cos(th)
        y = R*sin(th)

        d_u_x = -y*sin(x)
        d_u_y = cos(x)
        return d_u_x*cos(th) + d_u_y*sin(th)
