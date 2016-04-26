import numpy as np
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

    def get_boundary_coord(self, r1, th1):
        x1 = r1 * np.cos(th1)
        y1 = r1 * np.sin(th1)
        # Solve the boundary coordinate inverse problem
        #x1, y1 = sympy.symbols('x1 y1')

        th = sympy.symbols('th')
        x0 = self.r_expr * sympy.cos(th)
        y0 = self.r_expr * sympy.sin(th)

        # (Squared) distance between (x0, y0) and (x1, y1)
        d = (x1 - x0)**2 + (y1 - y0)**2
        d = d.subs(self.subs_dict)

        # Solve d'(th) = 0 to get minimum
        # TODO: What if d'(th) is never 0?
        d_d_th = sympy.diff(d, th)
        th0_list = sympy.solve(d_d_th, th)
        assert len(th0_list) != 0
        print('th0_list=', th0_list)

        min_th_diff = float('inf')

        for i in range(len(th0_list)):
            _th0 = float(th0_list[i])

            # Add 2pi to get _th0 in the interval [a, 2pi]
            if _th0 < self.a/2:
                _th0 += 2*np.pi

            # Choice the _th0 that is closest to th1
            if abs(th1 - _th0) < min_th_diff:
                th0 = _th0
                min_th_diff = abs(th1 - _th0)

        # Unsigned
        n = np.sqrt(float(d.subs(th, th0)))

        # Find the sign
        x0 = float(x0.subs(self.subs_dict).subs(th, th0))
        y0 = float(y0.subs(self.subs_dict).subs(th, th0))
        normal = self.eval_normal(th0)

        if np.dot(normal, [x1-x0, y1-y0]) < 0:
            n = -n

        print('n=',n)
        print('th0=', th0)
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
