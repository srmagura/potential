import numpy as np
from scipy.optimize import brentq
import sympy

from .problem import PizzaProblem


class Boundary:

    additional_params = {}

    def __init__(self, R, bet=None, k=None, a=PizzaProblem.a):
        self.a = a
        self.R = R

        if bet is None:
            self.set_bet(k)
        else:
            self.bet = bet

        self.subs_dict = {
            'R': self.R,
            'bet': self.bet,
            'a': self.a,
            'nu': np.pi / (2*np.pi-self.a)
        }

        self.subs_dict.update(self.additional_params)

        self.do_algebra()

    def set_bet(self, k):
        self.bet = self.bet0

    def do_algebra(self):
        """
        Setup sympy expressions and lambda functions for r, components of
        tangent vector, curvature, .etc.
        """
        # Get expression for r
        R, th, a, nu, bet = sympy.symbols('R th a nu bet')
        subs_dict = self.subs_dict

        self.r_expr = r = sympy.sympify(self.r_expr_str)

        self.eval_r = sympy.lambdify(th, r.subs(subs_dict))

        ## Setup expressions for finding tangent vector
        # Calculate d_x_th, d_y_th using the formula from:
        # http://tutorial.math.lamar.edu/Classes/CalcII/PolarTangents.aspx
        d_r_th = sympy.diff(r, th)
        d2_r_th = sympy.diff(d_r_th, th)

        x = r * sympy.cos(th)
        y = r * sympy.sin(th)
        d_x_th = sympy.diff(x, th)
        d_y_th = sympy.diff(y, th)

        self.eval_d_r_th = sympy.lambdify(th, d_r_th.subs(subs_dict))
        self.eval_d2_r_th = sympy.lambdify(th, d2_r_th.subs(subs_dict))

        self.eval_d_x_th = sympy.lambdify(th, d_x_th.subs(subs_dict))
        self.eval_d_y_th = sympy.lambdify(th, d_y_th.subs(subs_dict))

        ## Derivatives of the polar angle th wrt arclength s
        # Arclength in polar coordinates:
        # http://tutorial.math.lamar.edu/Classes/CalcII/PolarArcLength.aspx
        d_s_th = sympy.sqrt(r**2 + d_r_th**2)
        d2_s_th = sympy.diff(d_s_th, th)
        d3_s_th = sympy.diff(d2_s_th, th)
        d4_s_th = sympy.diff(d3_s_th, th)

        # These formulas come from the chain rule. See
        # https://en.wikipedia.org/wiki/Inverse_functions_and_differentiation#Higher_derivatives
        d_th_s = 1 / d_s_th
        d2_th_s = -d2_s_th * (d_th_s)**3
        d3_th_s = (-d3_s_th * (d_th_s)**4 -
            3 * d2_s_th * d2_th_s * (d_th_s)**2
        )
        d4_th_s = (-d4_s_th * (d_th_s)**5 -
            4 * d3_s_th * (d_th_s)**3 * d2_th_s -
            3 * (
                (d3_s_th * d_th_s * d2_th_s + d2_s_th * d3_th_s) *
                (d_th_s)**2 +
                2 * d2_s_th * (d2_th_s)**2 * d_th_s
            )
        )

        self.eval_d_th_s = sympy.lambdify(th, d_th_s.subs(subs_dict))
        self.eval_d2_th_s = sympy.lambdify(th, d2_th_s.subs(subs_dict))
        self.eval_d3_th_s = sympy.lambdify(th, d3_th_s.subs(subs_dict))
        self.eval_d4_th_s = sympy.lambdify(th, d4_th_s.subs(subs_dict))

        ## Curvature
        d2_r_th = sympy.diff(r, th, 2)

        # Polar form of curvature. Wikipedia has the numerator inside an
        # absolute value, but I took it out since our curvature is always
        # negative
        curv = - (r**2 + 2*d_r_th**2 - r*d2_r_th) / (r**2 + d_r_th**2)**(3/2)

        d_curv_th = sympy.diff(curv, th)
        d2_curv_th = sympy.diff(d_curv_th, th)

        self.eval_curv = sympy.lambdify(th, curv.subs(subs_dict))
        self.eval_d_curv_th = sympy.lambdify(th, d_curv_th.subs(subs_dict))
        self.eval_d2_curv_th = sympy.lambdify(th, d2_curv_th.subs(subs_dict))

    def eval_tangent(self, th):
        tangent = np.array((self.eval_d_x_th(th), self.eval_d_y_th(th)))
        return tangent / np.linalg.norm(tangent)

    def eval_normal(self, th):
        tangent = self.eval_tangent(th)
        return np.array((tangent[1], -tangent[0]))

    def get_boundary_coord(self, r1, th1):
        """
        Given a point with polar coordinates (r1, th1), find its
        coordinates (n, th) with respect to the boundary.
        """
        x1 = r1 * np.cos(th1)
        y1 = r1 * np.sin(th1)

        def eval_dist_deriv(th):
            # Let d = distance between (x0, y0) and (x1, y1)
            # This function returns the derivative of (1/2 d**2) wrt th
            r = self.eval_r(th)
            dr = self.eval_d_r_th(th)

            x0 = r * np.cos(th)
            y0 = r * np.sin(th)

            return ((x1 - x0) * (-dr*np.cos(th) + r*np.sin(th)) +
                (y1 - y0) * (-dr*np.sin(th) - r*np.cos(th)))

        # Optimization bounds
        diff = np.pi/7

        lbound = th1 - diff

        if lbound < self.a/2:
            lbound = self.a/2

        ubound = th1 + diff
        if ubound > 2*np.pi + self.a/2:
            ubound = 2*np.pi + self.a/2

        th0 = brentq(eval_dist_deriv, lbound, ubound, xtol=1e-16)

        # Get (absolute value of) n
        r = self.eval_r(th0)
        x0 = r * np.cos(th0)
        y0 = r * np.sin(th0)
        n = np.sqrt((x1-x0)**2 + (y1-y0)**2)

        # Find the sign
        normal = self.eval_normal(th0)
        if np.dot(normal, [x1-x0, y1-y0]) < 0:
            n = -n

        return n, th0

class Arc(Boundary):
    name = 'arc'
    r_expr_str = 'R'
    bet0 = 0.0

class OuterSine(Boundary):
    name = 'outer-sine'
    r_expr_str = 'R + bet*sin(nu*(th-a))'
    bet0 = 0.15

class InnerSine(Boundary):
    name = 'inner-sine'
    r_expr_str = 'R - bet*sin(nu*(th-a))'
    bet0 = 0.5

class Sine7(Boundary):
    name = 'sine7'
    r_expr_str = 'R + bet*sin(7*nu*(th-a))'
    bet0 = 0.025


cubic_C = (91*np.sqrt(91) - 136)*np.pi**3/2916

class Cubic(Boundary):
    name = 'cubic'
    r_expr_str = 'R + bet/C*(th-a)*(th-pi)*(th-2*pi)'

    bet0 = 0.075
    additional_params = {'C': cubic_C}


_boundary_cls = (Arc, OuterSine, InnerSine, Sine7, Cubic)

boundaries = {}
for boundary in _boundary_cls:
    boundaries[boundary.name] = boundary
