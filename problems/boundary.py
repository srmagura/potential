import numpy as np
from scipy.optimize import brentq
import sympy

from .problem import PizzaProblem


class Boundary:

    additional_params = {}

    def __init__(self, R, bet=None):
        self.a = PizzaProblem.a
        self.R = R

        if bet is None:
            self.bet = self.bet0
        else:
            self.bet = bet

        self.subs_dict = {
            'R': self.R,
            'bet': self.bet,
            'a': self.a,
            'nu': PizzaProblem.nu
        }

        self.subs_dict.update(self.additional_params)

        self.do_algebra()

    def do_algebra(self):
        """
        Setup sympy expressions and lambda functions for r, components of
        tangent vector, curvature, .etc.
        """
        # Get expression for r
        R, th, a, nu, bet = sympy.symbols('R th a nu bet')
        self.r_expr = r = sympy.sympify(self.r_expr_str)

        self.eval_r = sympy.lambdify(th, r.subs(self.subs_dict))

        ## Setup expressions for finding tangent vector
        # Calculate d_x_th, d_y_th using the formula from:
        # http://tutorial.math.lamar.edu/Classes/CalcII/PolarTangents.aspx
        d_r_th = sympy.diff(r, th)
        d_x_th = d_r_th * sympy.cos(th) - r * sympy.sin(th)
        d_y_th = d_r_th * sympy.sin(th) + r * sympy.cos(th)

        self.eval_d_r_th = sympy.lambdify(th, d_r_th.subs(self.subs_dict))
        self.eval_d_x_th = sympy.lambdify(th, d_x_th.subs(self.subs_dict))
        self.eval_d_y_th = sympy.lambdify(th, d_y_th.subs(self.subs_dict))

        ## Derivative of polar angle th wrt arclength s
        # Arclength in polar coordinates:
        # http://tutorial.math.lamar.edu/Classes/CalcII/PolarArcLength.aspx
        d_th_s = 1 / sympy.sqrt(r**2 + d_r_th**2)
        self.eval_d_th_s = sympy.lambdify(th, d_th_s.subs(self.subs_dict))

        ## Curvature
        d2_r_th = sympy.diff(r, th, 2)

        curv = - abs(r**2 + 2*d_r_th**2 - r*d2_r_th) / (r**2 + d_r_th**2)**(3/2)
        self.eval_curv = sympy.lambdify(th, curv.subs(self.subs_dict))

        ## Lame coefficient Hs (TODO delete?)
        n = sympy.symbols('n')
        Hs = 1 - n*curv

        # Derivative of 1/Hs wrt s
        d_Hs1_s = sympy.diff(1/Hs, th) * d_th_s

        self.eval_Hs = sympy.lambdify((n, th), Hs.subs(self.subs_dict))
        self.eval_d_Hs1_s = sympy.lambdify((n, th), d_Hs1_s.subs(self.subs_dict))

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
        diff = np.pi/6

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
    bet0 = 0

class OuterSine(Boundary):
    name = 'outer-sine'
    r_expr_str = 'R + bet*sin(nu*(th-a))'
    bet0 = 0.25

class InnerSine(Boundary):
    name = 'inner-sine'
    r_expr_str = 'R - bet*sin(nu*(th-a))'
    bet0 = 0.5

class Sine7(Boundary):
    name = 'sine7'
    r_expr_str = 'R + bet*sin(7*nu*(th-a))'
    bet0 = 0.05


cubic_C = (91*np.sqrt(91) - 136)*np.pi**3/2916

class Cubic(Boundary):
    name = 'cubic'
    r_expr_str = 'R + bet/C*(th-a)*(th-pi)*(th-2*pi)'

    bet0 = 0.025*cubic_C
    additional_params = {'C': cubic_C}

_boundary_cls = (Arc, OuterSine, InnerSine, Sine7, Cubic)

boundaries = {}
for boundary in _boundary_cls:
    boundaries[boundary.name] = boundary
