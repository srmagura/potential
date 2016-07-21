import math
import numpy as np


class SolverExtend:
    """
    Extension procedures available to all Solvers.
    """

    def extend_polar(self, **kwargs):
        """
        Homogeneous extension from an arbitrary boundary curve that
        is parameterized by the polar angle th.
        """
        curv = kwargs['curv']
        d_curv_th = kwargs['d_curv_th']
        d2_curv_th = kwargs['d2_curv_th']

        d_th_s = kwargs['d_th_s']
        d2_th_s = kwargs['d2_th_s']
        d3_th_s = kwargs['d3_th_s']
        d4_th_s = kwargs['d4_th_s']

        d_xi0_th = kwargs['d_xi0_th']
        d_xi1_th = kwargs['d_xi1_th']
        d2_xi0_th = kwargs['d2_xi0_th']
        d2_xi1_th = kwargs['d2_xi1_th']
        d3_xi0_th = kwargs['d3_xi0_th']
        d4_xi0_th = kwargs['d4_xi0_th']

        def convert_d(d):
            return d * d_th_s

        def convert_d2(d, d2):
            return d2 * d_th_s**2 + d * d2_th_s

        def convert_d4(d, d2, d3, d4):
            return (d_th_s**4 * d4 +
                6 * d_th_s**2 * d2_th_s * d3 +
                4 * d_th_s * d2 * d3_th_s +
                3 * d2_th_s**2 * d2 +
                d4_th_s * d
            )

        return self.extend_arbitrary(
            n=kwargs['n'],
            xi0=kwargs['xi0'],
            xi1=kwargs['xi1'],
            d_xi0_s=convert_d(d_xi0_th),
            d_xi1_s=convert_d(d_xi1_th),
            d2_xi0_s=convert_d2(d_xi0_th, d2_xi0_th),
            d2_xi1_s=convert_d2(d_xi1_th, d2_xi1_th),
            d4_xi0_s=convert_d4(d_xi0_th, d2_xi0_th, d3_xi0_th, d4_xi0_th),
            curv=curv,
            d_curv_s=convert_d(d_curv_th),
            d2_curv_s=convert_d2(d_curv_th, d2_curv_th)
        )


    def extend_arbitrary(self, **kwargs):
        """
        Homogeneous extension from an arbitrary boundary curve.

        Computes a Taylor expansion with five derivatives.
        """
        n = kwargs['n']
        curv = kwargs['curv']
        d_curv_s = kwargs['d_curv_s']
        d2_curv_s = kwargs['d2_curv_s']

        xi0 = kwargs['xi0']
        xi1 = kwargs['xi1']
        d_xi0_s = kwargs['d_xi0_s']
        d_xi1_s = kwargs['d_xi1_s']
        d2_xi0_s = kwargs['d2_xi0_s']
        d2_xi1_s = kwargs['d2_xi1_s']
        d4_xi0_s = kwargs['d4_xi0_s']

        k = self.k

        derivs = []
        derivs.append(xi0)
        derivs.append(xi1)

        derivs.append(-k**2 * xi0 + curv * xi1 - d2_xi0_s)

        derivs.append((curv**2 - k**2) * xi1 +
            curv*derivs[2] - d_curv_s * d_xi0_s -
            2*curv*d2_xi0_s - d2_xi1_s
        )

        d4_v_n2_s2 = (d2_curv_s * xi1 - k**2 * d2_xi0_s - d4_xi0_s +
            2*d_curv_s*d_xi1_s + curv*d2_xi1_s)

        derivs.append(2*curv**3 * xi1 +
            (2*curv**2 - k**2) * derivs[2] +
            curv * derivs[3] -
            6 * curv * d_curv_s * d_xi0_s -
            6 * curv**2 * d2_xi0_s -
            2 * d_curv_s * d_xi1_s -
            4 * curv * d2_xi1_s -
            d4_v_n2_s2
        )

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * n**l

        return v

    def inhomo_extend_polar(self, **kwargs):
        """
        Inhomogeneous extension that converts derivatives wrt r and th
        to derivatives wrt arclength s
        """
        d_f_th = kwargs.pop('d_f_th')
        d2_f_th = kwargs.pop('d2_f_th')

        d_f_r = kwargs.pop('d_f_r')
        d2_f_r = kwargs.pop('d2_f_r')
        d2_f_r_th = kwargs.pop('d2_f_r_th')

        d_th_s = kwargs.pop('d_th_s')
        d2_th_s = kwargs.pop('d2_th_s')

        d_r_th = kwargs.pop('d_r_th')
        d2_r_th = kwargs.pop('d2_r_th')

        d_r_s = d_r_th * d_th_s
        d2_r_s = d2_r_th * d_th_s**2 + d_r_th * d2_th_s

        d2_f_th_s = d2_f_th * d_th_s + d2_f_r_th * d_r_s
        d2_f_r_s = d2_f_r_th * d_th_s + d2_f_r * d_r_s

        d2_f_s = (
            d2_f_th_s * d_th_s
            + d_f_th * d2_th_s
            + d2_f_r_s * d_r_s
            + d_f_r * d2_r_s
        )

        kwargs['d2_f_s'] = d2_f_s
        return self.inhomo_extend_arbitrary(**kwargs)

    def inhomo_extend_arbitrary(self, **kwargs):
        """ Arbitrary inhomogeneous extension  """
        k = self.k

        n = kwargs['n']
        f = kwargs['f']

        d_f_n = kwargs['d_f_n']
        d2_f_n = kwargs['d2_f_n']
        d2_f_s = kwargs['d2_f_s']

        curv = kwargs['curv']

        derivs = [0, 0, f]

        derivs.append(d_f_n + curv * derivs[2])

        derivs.append(
            d2_f_n
            + (2*curv**2 - k**2) * derivs[2]
            + curv * derivs[3]
            - d2_f_s
        )

        #print('------------------------')
        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * n**l

        return v
