import numpy as np
from scipy.optimize import minimize_scalar
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
        Parse the string for the r expression. Calculate d_x_th, d_y_th
        using the formula from:
        http://tutorial.math.lamar.edu/Classes/CalcII/PolarTangents.aspx
        """

        # Setup expression for evaluating r
        R, th, a, nu, bet = sympy.symbols('R th a nu bet')
        self.r_expr = r_expr = sympy.sympify(self.r_expr_str)

        self.r_lambda = sympy.lambdify(th, r_expr.subs(self.subs_dict))

        # Setup expressions for finding tangent vector
        d_r_th_expr = sympy.diff(r_expr, th)
        d_x_th_expr = d_r_th_expr * sympy.cos(th) - r_expr * sympy.sin(th)
        d_y_th_expr = d_r_th_expr * sympy.sin(th) + r_expr * sympy.cos(th)

        self.d_x_th_lambda = sympy.lambdify(th,
            d_x_th_expr.subs(self.subs_dict))
        self.d_y_th_lambda = sympy.lambdify(th,
            d_y_th_expr.subs(self.subs_dict))



    def eval_r(self, th):
        return self.r_lambda(th)

    def eval_tangent(self, th):
        tangent = np.array((self.d_x_th_lambda(th), self.d_y_th_lambda(th)))
        return tangent / np.linalg.norm(tangent)

    def eval_normal(self, th):
        tangent = self.eval_tangent(th)
        return np.array((tangent[1], -tangent[0]))

    def get_boundary_coord(self, r1, th1, extended):
        x1 = r1 * np.cos(th1)
        y1 = r1 * np.sin(th1)

        def eval_distance2(th):
            # (Squared) distance between (x0, y0) and (x1, y1)

            r = self.eval_r(th)
            x0 = r * np.cos(th)
            y0 = r * np.sin(th)

            return (x1 - x0)**2 + (y1 - y0)**2

        # Optimization bound
        diff = np.pi/6
        bounds = [th1 - diff, th1 + diff]

        if not extended:
            # Restrict optimization to the non-extended boundary
            if bounds[0] < self.a:
                bounds[0] = self.a
            if bounds[1] > 2*np.pi:
                bounds[1] = 2*np.pi

        result = minimize_scalar(eval_distance2,
            bounds=bounds,
            method='bounded'
        )

        th0 = result.x

        # Unsigned
        n = np.sqrt(eval_distance2(th0))

        # Find the sign
        r = self.eval_r(th0)
        x0 = r * np.cos(th0)
        y0 = r * np.sin(th0)

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
    bet0 = 0.15


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