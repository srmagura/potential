import numpy as np
import sympy

from .problem import PizzaProblem


class Boundary:

    additional_params = {}

    def __init__(self, R, bet=None):
        self.R = R

        if bet is None:
            self.bet = self.bet0
        else:
            self.bet = bet

        self.subs_dict = {
            'R': self.R,
            'bet': self.bet,
            'a': PizzaProblem.a,
            'nu': PizzaProblem.nu
        }

        self.subs_dict.update(self.additional_params)

        self.parse_r_expr_str()

    def parse_r_expr_str(self):
        R, th, a, nu, bet = sympy.symbols('R th a nu bet')
        self.r_expr = sympy.sympify(self.r_expr_str)

        subs_r_expr = self.r_expr.subs(self.subs_dict)
        self.r_lambda = sympy.lambdify(th, subs_r_expr)

    def eval_r(self, th):
        return self.r_lambda(th)


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
