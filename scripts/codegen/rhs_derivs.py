"""
Generate Python code for evaluating a problem's RHS and its derivatives.

problems.SympyProblem provides similar functionality and is more convenient,
but slightly slower.
"""

from sympy import *

k = symbols('k')
x, y = symbols('x y')
r, th = symbols('r th')

def print_code(signature, return_value):
    ind = ' '*4
    print()
    print(ind + 'def {}:'.format(signature))
    print(2*ind + 'k = self.k')
    print(2*ind + 'return {}'.format(return_value))


def find_derivs(u):
    print_code('eval_expected(self, x, y)', u)

    f = diff(u, x, 2) + diff(u, y, 2) + k**2 * u
    print_code('eval_f(self, x, y)', f)

    f_polar = f.subs(dict(x=r*cos(th), y=r*sin(th)))
    find_f_polar_derivs(f_polar)

    find_f_total_derivs(f)


def find_derivs_polar(u):
    print_code('eval_expected_polar(self, r, th)', u)

    f = diff(u, r, 2) + 1/r * diff(u, r) + 1/r**2 * diff(u, th, 2) +  k**2 * u
    print_code('eval_f_polar(self, r, th)', f)

    find_f_polar_derivs(f)
    #TODO: find_f_total_derivs


def find_f_polar_derivs(f_polar):
    print_code('eval_d_f_r(self, r, th)', diff(f_polar, r))
    print_code('eval_d2_f_r(self, r, th)', diff(f_polar, r, 2))

    print_code('eval_d_f_th(self, r, th)', diff(f_polar, th))
    print_code('eval_d2_f_th(self, r, th)', diff(f_polar, th, 2))

    print_code('eval_d2_f_r_th(self, r, th)', diff(f_polar, r, th))


def find_f_total_derivs(f):
    grad = (diff(f, x), diff(f, y))
    grad_np = 'np.array({})'.format(grad)
    print_code('eval_grad_f(self, x, y)', grad_np)

    hessian = (
        (diff(f, x, x), diff(f, x, y)),
        (diff(f, y, x), diff(f, y, y))
    )
    hessian_np = 'np.array({})'.format(hessian)
    print_code('eval_hessian_f(self, x, y)', hessian_np)
