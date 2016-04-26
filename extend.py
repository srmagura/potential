import math
import numpy as np

class SolverExtend:
    """
    Extension procedures available to all Solvers.
    """

    def extend_arbitrary(self, n, xi0, xi1,
        d2_xi0_th, d2_xi1_th, d4_xi0_th):
        """
        Homogeneous extension from the outside boundary.

        Computes a Taylor expansion with five derivatives. See [R1]
        section 4.2.

        n -- signed distance from boundary
        """

        R = self.R
        k = self.k

        derivs = []
        derivs.append(xi0)


        derivs.append(xi1)
        #derivs.append(-xi1 / R - d2_xi0_th / R**2 - k**2 * xi0)

        '''if self.extension_order > 3:
            derivs.append(2 * xi1 / R**2 + 3 * d2_xi0_th / R**3 -
                d2_xi1_th / R**2 + k**2 / R * xi0 - k**2 * xi1)

            derivs.append(-6 * xi1 / R**3 +
                (2*k**2 / R**2 - 11 / R**4) * d2_xi0_th +
                6 * d2_xi1_th / R**3 + d4_xi0_th / R**4 -
                (3*k**2 / R**2 - k**4) * xi0 +
                2 * k**2 / R * xi1)
        '''

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * n**l

        return v

    def calc_inhomo_circle(self, r, th):
        """
        Inhomogeneous extension from the circle/arc, including the
        calculation of the necessary derivatives of f.
        """
        p = self.problem
        if p.homogeneous:
            return 0

        R = self.R
        x = R * np.cos(th)
        y = R * np.sin(th)

        f = p.eval_f(x, y)

        if self.extension_order > 3:
            d_f_r = p.eval_d_f_r(R, th)
            d2_f_r = p.eval_d2_f_r(R, th)
            d2_f_th = p.eval_d2_f_th(R, th)
        else:
            d_f_r = 0
            d2_f_r = 0
            d2_f_th = 0

        return self.extend_inhomo_circle(
            r, f, d_f_r, d2_f_r, d2_f_th)

    def extend_inhomo_circle(self, r, f, d_f_r, d2_f_r, d2_f_th):
        """ Inhomogeneous extension from the circle/arc. """
        R = self.R
        k = self.k

        derivs = [0, 0, f]

        # Don't need any more derivatives if using second order scheme
        if self.extension_order > 3:
            derivs.extend([
                d_f_r - f / R,
                d2_f_r - d2_f_th / R**2 - d_f_r / R + (3/R**2 - k**2) * f
            ])

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * (r - R)**l

        return v
