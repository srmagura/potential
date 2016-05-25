import itertools as it
import enum

import numpy as np
from numpy import sin, cos
import math

import copy

from domain_util import cart_to_polar
from .multivalue import Multivalue

EType = enum.Enum('EType', 'left standard right')

def get_deriv_name(order, index):
    if order == 0:
        return 'xi{}'.format(index)
    elif order == 1:
        return 'd_xi{}_arg'.format(index)
    else:
        return'd{}_xi{}_arg'.format(order, index)


class PsExtend:

    taylor_n_terms = 6

    def get_etype(self, sid, i, j):
        a = self.a
        R = self.R

        x, y = self.get_coord(i, j)

        if sid == 0:
            n, th = self.boundary_coord_cache[(i, j)]

            if th > 2*np.pi:
                return EType.right
            elif th < a:
                return EType.left
            else:
                return EType.standard

        elif sid == 1:
            if x < 0:
                return EType.left
            elif x > R:
                return EType.right
            else:
                return EType.standard

        elif sid == 2:
            x1, y1 = self.get_radius_point(2, x, y)
            if y1 < 0:
                return EType.left
            elif y1 > R*np.sin(a):
                return EType.right
            else:
                return EType.standard

    def extend_radius(self, **kwargs):
        return self.extend_arbitrary(
            n=kwargs['n'],
            curv=0,
            d_curv_s=0,
            d2_curv_s=0,
            xi0=kwargs['xi0'],
            xi1=kwargs['xi1'],
            d_xi0_s=kwargs['d_xi0_arg'],
            d_xi1_s=kwargs['d_xi1_arg'],
            d2_xi0_s=kwargs['d2_xi0_arg'],
            d2_xi1_s=kwargs['d2_xi1_arg'],
            d4_xi0_s=kwargs['d4_xi0_arg'],
        )

    def _ext_calc_xi_derivs(self, deriv_types, arg, sid,
        human_readable_keys=True, **kwargs):
        """
        Calculate derivatives of xi0, xi1, or a basis function.

        Should not be accessed directly. Call ext_calc_xi_derivs or
        ext_calc_B_derivs instead.

        deriv_types -- iterable of 2-tuples of the form
            (order, index) where order is an integer >= 0 that
            indicates which derivative to take, and index=0,1
            specifies xi0 or xi1
        arg -- location on the boundary where derivative is evaluated
        JJ -- ID of basis function to calculate derivatives of.
            If None,

        Returns a dictionary containing all of the requested derivatives.
        """
        if kwargs['JJ'] is not None:
            JJ = kwargs['JJ']
            basis_index = kwargs['basis_index']

            JJ_list = [JJ]

            c = [{}, {}]
            c[basis_index][JJ] = 1

            if basis_index == 0:
                other_basis_index = 1
            elif basis_index == 1:
                other_basis_index = 0

            c[other_basis_index][JJ] = 0

        else:
            JJ_list = range(len(self.B_desc))
            c = [self.c0, self.c1]

        derivs = {}

        for (order, index) in deriv_types:
            deriv = 0
            for JJ in JJ_list:
                dn_B_arg = self.eval_dn_B_arg(order, JJ, arg, sid)
                deriv += c[index][JJ] * dn_B_arg

            if human_readable_keys:
                key = get_deriv_name(order, index)
            else:
                key = (order, index)

            derivs[key] = deriv

        return derivs

    def ext_calc_xi_derivs(self, deriv_types, arg, sid,
        human_readable_keys=True, **kwargs):
        """
        Calculate derivatives of xi0 and xi1.

        deriv_types -- iterable of 2-tuples of the form
            (order, index) where order is an integer >= 0 that
            indicates which derivative to take, and index=0,1
            specifies xi0 or xi1
        arg -- location on the boundary where derivative is evaluated
        human_readable_keys -- If True, key for second derivative
            of xi0 will be 'd2_xi0_arg'. If False, the key will be the
            tuple (2, 0).

        Returns a dictionary containing all of the requested derivatives.
        """
        return self._ext_calc_xi_derivs(deriv_types, arg, sid,
            human_readable_keys=human_readable_keys, **kwargs)

    def do_extend_taylor(self, i, j, arg, taylor_sid, delta_arg, n=None,
        **kwargs):
        """
        Use a Taylor expansion to extend the boundary data from the
        physical boundary to the "extended boundary". Then extend to
        the grid node (i, j) using the equation-based extension.

        **kwargs:
            JJ: if extending a single basis function, contains
                the ID of that basis function. If extending the full
                boundary data, JJ should be None.
            basis_index: indicates whether this is a Dirichlet
                or Neumann basis function. Value does not matter if
                JJ is None.
        """
        # Build the dictionary xi_derivs0, which will hold the
        # tangential derivatives at the "starting point" of the
        # Taylor expansion. Keys of the dictionary are of the
        # form (order, index)

        deriv_types = list(it.product(range(self.taylor_n_terms), (0, 1)))
        xi_derivs0 = self.ext_calc_xi_derivs(deriv_types, arg, taylor_sid,
            human_readable_keys=False, **kwargs)

        # To hold the results of the Taylor expansion. Tangential
        # derivatives, with respect to the argument (r or th) for
        # this section of the boundary
        xi_derivs = {}

        for index in (0, 1):
            for m in range(self.taylor_n_terms):
                # Compute m-th derivative by Taylor series.
                # Taylor series will have (taylor_n_terms-m) terms.
                xi_derivs[(m, index)] = 0

                for l in range(m, self.taylor_n_terms):
                    # Contribution of the l-th derivative to the m-th
                    # derivative
                    fac = delta_arg**(l-m) / math.factorial(l-m)
                    xi_derivs[(m, index)] += xi_derivs0[(l, index)] * fac

        # Convert to human readable dictionary keys
        for key in list(xi_derivs.keys()):
            order, index = key
            deriv_name = get_deriv_name(order, index)
            xi_derivs[deriv_name] = xi_derivs[key]
            del xi_derivs[key]

        if taylor_sid == 0:
            v = self.do_extend_0(i, j, xi_derivs)

            # This is only an approximation for perturbed boundaries,
            # but it shouldn't really matter.
            n, th = self.boundary_coord_cache[(i, j)]
            elen = abs(delta_arg)*self.R + abs(n)

        elif taylor_sid in {1, 2}:
            v = self.extend_radius(
                n=n,
                **xi_derivs
            )

            elen = abs(delta_arg) + abs(n)

        return {'elen': elen, 'value': v}

    def do_extend_0(self, i, j, derivs):
        n, th = self.boundary_coord_cache[(i, j)]

        value = self.extend_polar(
            n=n,
            curv=self.boundary.eval_curv(th),
            d_curv_th=self.boundary.eval_d_curv_th(th),
            d2_curv_th=self.boundary.eval_d2_curv_th(th),
            d_th_s=self.boundary.eval_d_th_s(th),
            d2_th_s=self.boundary.eval_d2_th_s(th),
            d3_th_s=self.boundary.eval_d3_th_s(th),
            d4_th_s=self.boundary.eval_d4_th_s(th),
            xi0=derivs['xi0'],
            xi1=derivs['xi1'],
            d_xi0_th=derivs['d_xi0_arg'],
            d_xi1_th=derivs['d_xi1_arg'],
            d2_xi0_th=derivs['d2_xi0_arg'],
            d2_xi1_th=derivs['d2_xi1_arg'],
            d3_xi0_th=derivs['d3_xi0_arg'],
            d4_xi0_th=derivs['d4_xi0_arg'],
        )

        return value

    """
    Documentation for the functions:
    do_extend_[SID]_[left/right/standard]

    Args:
        i, j: grid node, "destination" of the extension
        **kwargs:
            JJ: if extending a single basis function, contains
                the ID of that basis function. If extending the full
                boundary data, JJ should be None.
            basis_index: indicates whether this is a Dirichlet
                or Neumann basis function. Value does not matter if
                JJ is None.
    """

    def do_extend_0_standard(self, i, j, **kwargs):
        n, th = self.boundary_coord_cache[(i, j)]

        deriv_types = (
            (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (4, 0)
        )

        derivs = self.ext_calc_xi_derivs(deriv_types, th, 0, **kwargs)
        return {'elen': abs(n), 'value': self.do_extend_0(i, j, derivs)}

    def do_extend_0_left(self, i, j, **kwargs):
        n, th0 = self.boundary_coord_cache[(i, j)]

        return self.do_extend_taylor(i, j, arg=self.a,
            taylor_sid=0, delta_arg=th0-self.a, **kwargs)

    def do_extend_0_right(self, i, j, **kwargs):
        n, th0 = self.boundary_coord_cache[(i, j)]

        return self.do_extend_taylor(i, j, arg=2*np.pi,
            taylor_sid=0, delta_arg=th0-2*np.pi, **kwargs)

    def do_extend_1_standard(self, i, j, **kwargs):
        x, y = self.get_coord(i, j)

        deriv_types = (
            (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (4, 0),
        )

        derivs = self.ext_calc_xi_derivs(deriv_types, x, 1, **kwargs)
        value = self.extend_radius(n=y, **derivs)

        return {'elen': abs(y), 'value': value}

    def do_extend_1_left(self, i, j, **kwargs):
        x, y = self.get_coord(i, j)
        return self.do_extend_taylor(i, j, arg=0, taylor_sid=1,
            delta_arg=x, n=y, **kwargs)

    def do_extend_1_right(self, i, j, **kwargs):
        x, y = self.get_coord(i, j)
        R = self.R
        return self.do_extend_taylor(i, j, arg=R, taylor_sid=1,
            delta_arg=x-R, n=y, **kwargs)

    def do_extend_2_standard(self, i, j, **kwargs):
        x, y = self.get_coord(i, j)
        x0, y0 = self.get_radius_point(2, x, y)

        param_r = cart_to_polar(x0, y0)[0]

        deriv_types = (
            (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (4, 0),
        )
        derivs = self.ext_calc_xi_derivs(deriv_types, param_r, 2, **kwargs)
        n = self.signed_dist_to_radius(2, x, y)

        value = self.extend_radius(n=n, **derivs)

        return {'elen': abs(n), 'value': value}

    def do_extend_2_left(self, i, j, **kwargs):
        x, y = self.get_coord(i, j)
        x1, y1 = self.get_radius_point(2, x, y)

        return self.do_extend_taylor(i, j, arg=0, taylor_sid=2,
            delta_arg=-np.sqrt(x1**2 + y1**2),
            n=self.signed_dist_to_radius(2, x, y),
            **kwargs
        )

    def do_extend_2_right(self, i, j, **kwargs):
        R = self.R
        a = self.a

        x, y = self.get_coord(i, j)

        x0 = R*cos(a)
        y0 = R*sin(a)
        x1, y1 = self.get_radius_point(2, x, y)

        return self.do_extend_taylor(i, j, arg=R, taylor_sid=2,
            delta_arg=np.sqrt((x1 - x0)**2 + (y1 - y0)**2),
            n=self.signed_dist_to_radius(2, x, y),
            **kwargs
        )

    def mv_extend_boundary(self, JJ=None, basis_index=None,
        homogeneous_only=False):
        """
        Extend the boundary data to the discrete boundary. Returns a
        Multivalue object (multiple-valued grid function).
        """
        R = self.R
        k = self.problem.k

        mv_ext = Multivalue(self.union_gamma)

        for sid in range(3):
            gamma = self.all_gamma[sid]

            for l in range(len(gamma)):
                kwargs = {'JJ': JJ, 'basis_index': basis_index}

                i, j = gamma[l]
                etype = self.get_etype(sid, i, j)

                result = None

                if sid == 0:
                    if etype == EType.standard:
                        result = self.do_extend_0_standard(i, j, **kwargs)
                    elif etype == EType.left:
                        result = self.do_extend_0_left(i, j, **kwargs)
                    elif etype == EType.right:
                        result = self.do_extend_0_right(i, j, **kwargs)

                elif sid == 1:
                    if etype == EType.standard:
                        result = self.do_extend_1_standard(i, j, **kwargs)
                    elif etype == EType.left:
                        result = self.do_extend_1_left(i, j, **kwargs)
                    elif etype == EType.right:
                        result = self.do_extend_1_right(i, j, **kwargs)

                elif sid == 2:
                    if etype == EType.standard:
                        result = self.do_extend_2_standard(i, j, **kwargs)
                    elif etype == EType.left:
                        result = self.do_extend_2_left(i, j, **kwargs)
                    elif etype == EType.right:
                        result = self.do_extend_2_right(i, j, **kwargs)

                result['setype'] = (sid, etype)
                mv_ext[(i, j)].append(result)

        # Don't add the inhomogeneous part if
        # - this is a basis function, or
        # - this is a homogeneous problem, or
        # - this function received the 'homogeneous_only' option, which is
        #   used for debugging

        if(JJ is not None or self.problem.homogeneous or homogeneous_only):
            return mv_ext
        else:
            return mv_ext + self.mv_extend_inhomo_f()

    def extend_boundary(self, **kwargs):
        """
        Extend the boundary data to the discrete boundary. Returns
        a single-valued function defined on the discrete boundary.
        """
        mv_ext = self.mv_extend_boundary(**kwargs)
        return mv_ext.reduce()
