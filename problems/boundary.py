import numpy as np
import sympy

from .problem import PizzaProblem


class Boundary:

    def __init__(self, R):
        self.R = R
        self.subs_dict = {
            'R': self.R,
            'a': PizzaProblem.a,
            'nu': PizzaProblem.nu
        }

        self.parse_r_expr_str()

    def parse_r_expr_str(self):
        R, th, a, nu = sympy.symbols('R th a nu')
        self.r_expr = sympy.sympify(self.r_expr_str)

        subs_r_expr = self.r_expr.subs(self.subs_dict)
        self.r_lambda = sympy.lambdify(th, subs_r_expr)

    def eval_r(self, th):
        return self.r_lambda(th)


class Arc(Boundary):
    name = 'arc'
    r_expr_str = 'R'

class OuterSine(Boundary):
    name = 'outer-sine'
    r_expr_str = 'R + 0.5*sin(nu*(th-a))'

class InnerSine(Boundary):
    name = 'inner-sine'
    r_expr_str = 'R - 0.5*sin(nu*(th-a))'

class Sine7(Boundary):
    name = 'sine7'
    r_expr_str = 'R + 0.15*sin(7*nu*(th-a))'

class Cubic(Boundary):
    name = 'cubic'
    r_expr_str = 'R + .025*(th-a)*(th-pi)*(th-2*pi)'


_boundary_cls = (Arc, OuterSine, InnerSine, Sine7, Cubic)

boundaries = {}
for boundary in _boundary_cls:
    boundaries[boundary.name] = boundary
