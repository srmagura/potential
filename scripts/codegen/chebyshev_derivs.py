"""
Generate Python code that computes the derivatives of the Chebyshev
polynomials.
"""
from sympy import *

J, t = symbols('J t')
T = cos(J * acos(t))

for n in range(1, 6):
    dn_T_t = diff(T, t, n)
    print('d{}_T_t ='.format(n))
    print(dn_T_t)
    print()
