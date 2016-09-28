""" Apply the Helmholtz operator """
from sympy import *

r, th, k, x, y = symbols('r th k x y')

def apply_helmholtz_op(u):
    d_u_r = diff(u, r)
    return diff(d_u_r, r) + d_u_r / r + diff(u, th, 2) / r**2 + k**2 * u

u = r**8
print('Input:')
print(u)

print()
print('Output:')
print(apply_helmholtz_op(u))

print()
print('Grad:')
u = u.subs(r, sqrt(x**2+y**2))
print((diff(u, x), diff(u, y)))
