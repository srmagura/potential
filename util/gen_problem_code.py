from sympy import *

def find_derivatives(u):
    print('def eval_expected(self, x, y):')
    print(rs + str(u))

    k = Symbol('self.k') 

    f = diff(u, x, 2) + diff(u, y, 2) + k**2 * u
    print()
    print('def eval_f(self, x, y):')
    print(rs + str(f))

    f_polar = f.subs(dict(x=r*cos(th), y=r*sin(th)))
    find_f_derivatives(f_polar)

def find_derivatives_polar(u):
    print('def eval_expected_polar(self, r, th):')
    print(rs + str(u))

    k = Symbol('self.k') 

    f = diff(u, r, 2) + 1/r * diff(u, r) + 1/r**2 * diff(u, th, 2) +  k**2 * u
    print()
    print('def eval_f_polar(self, r, th):')
    print(rs + str(f))

    find_f_derivatives(f)

def find_f_derivatives(f_polar):
    print()
    print('def eval_d_f_r(self, r, th):')
    print(rs + str(diff(f_polar, r)))

    print()
    print('def eval_d2_f_r(self, r, th):')
    print(rs + str(diff(f_polar, r, 2)))

    print()
    print('def eval_d2_f_th(self, r, th):')
    print(rs + str(diff(f_polar, th, 2)))

rs = '    return '
x, y = symbols('x y')
r, th = symbols('r th')

u = cos(r)
find_derivatives_polar(u)
