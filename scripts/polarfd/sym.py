from sympy import *
import scipy.special

def do_lapacian(u):
    return diff(u, r, 2) + diff(u, r) / r + diff(u, th, 2) / r**2 + k**2 * u

_k = 1

a = pi/6
nu = pi / (2*pi - a)

k, r, th = symbols('k r th')
kr2 = k*r/2

v_asympt0 = 0
for l in range(3):
    x = 3/11 + l + 1
    v_asympt0 += (-1)**l/(factorial(l)*gamma(x)) * kr2**(2*l)

v_asympt0 *= kr2**(sympify('3/11'))

v_asympt = v_asympt0 * sin(nu/2*(th-a))

f = -do_lapacian(v_asympt)
f = f.subs(k, _k)

modules = ('numpy', {
    'gamma': scipy.special.gamma,
    'besselj': scipy.special.jv
})

eval_v_asympt = lambdify((r, th), v_asympt.subs(k,_k), modules=modules)
eval_f = lambdify((r, th), f, modules=modules)
eval_d_f_r = lambdify((r, th), diff(f, r), modules=modules)
eval_d2_f_r = lambdify((r, th), diff(f, r, 2), modules=modules)
eval_d2_f_th = lambdify((r, th), diff(f, th, 2), modules=modules)
