import numpy as np
from scipy.special import jv, jvp

from .problem import PizzaProblem


class SingTest(PizzaProblem):

    k = 1
    m = 1

    homogeneous = True
    expected_known = True

    def eval_expected_polar(self, r, th):
        a = self.a
        nu = self.nu
        k = self.k
        m = self.m

        return jv(m*nu, k*r) * np.sin(m*nu*(th-a))

    def eval_bc_extended(self, arg, sid):
        r, th = self.arg_to_polar(arg, sid)

        if sid == 0:
            return self.eval_expected_polar(r, th)
        else:
            return 0

    def eval_d_u_outwards(self, arg, sid):
        a = self.a
        nu = self.nu
        k = self.k
        m = self.m

        r, th = self.arg_to_polar(arg, sid)

        if sid == 0:
            return k * jvp(m*nu, k*r) * np.sin(m*nu*(th-a))

        d_u_th = jv(m*nu, k*r) * m * nu * np.cos(m*nu*(th-a))
        d_u_outwards = d_u_th / r
        if sid == 1:
            return d_u_outwards
        elif sid == 2:
            return -d_u_outwards
