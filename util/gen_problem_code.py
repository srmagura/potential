from sympy import *

def find_derivatives(u):
    rs = '    return '
    print('def eval_expected(self, x, y):')
    print(rs + str(u))

    k = Symbol('self.k') 

    f = diff(u, x, 2) + diff(u, y, 2) + k**2 * u
    print()
    print('def eval_f(self, x, y):')
    print(rs + str(f))

    r, th = symbols('r th')
    f_polar = f.subs(dict(x=r*cos(th), y=r*sin(th)))
    print()
    print('def eval_d_f_r(self, r, th):')
    print(rs + str(diff(f_polar, r)))

    print()
    print('def eval_d2_f_r(self, r, th):')
    print(rs + str(diff(f_polar, r, 2)))

    print()
    print('def eval_d2_f_th(self, r, th):')
    print(rs + str(diff(f_polar, th, 2)))
