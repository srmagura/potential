from sympy import *
import numpy as np
import random

import problems.jump as jump
import problems.bessel as bessel

n_trials = 100
th_interval = (np.pi/6+.01, 2*np.pi-.01)

def do_test(u, manual_f_expr):
    args = symbols('k R r th')
    k, R, r, th = args

    sympy_f_expr = diff(u, r, 2) + diff(u, r)/r + diff(u, th, 2)/r**2 + k**2*u
    sympy_f_lambda = lambdify(args, sympy_f_expr)

    manual_f_lambda = lambdify(args, manual_f_expr)

    error = np.zeros(n_trials)

    for i in range(n_trials):
        k = random.uniform(.5, 3)
        R = random.uniform(1, 3)
        r = random.uniform(.01, 1)
        
        th = random.uniform(*th_interval)
    
        sympy_f = sympy_f_lambda(k, R, r, th)
        manual_f = manual_f_lambda(k, R, r, th)
    
        error[i] = abs(sympy_f - manual_f)
    
    return max(error)
           

all_tests = {
    'jump-reg0': (
        -jump.get_u04_expr(),
        jump.get_reg_f_expr(),
    ),
    
    'jump-reg1': (
        -jump.get_u04_sng_expr(),
        jump.get_reg_f_sng_expr()
    ),
    
    'bessel-reg': (
        -bessel.get_u_asympt_expr(),
        bessel.get_reg_f_expr()
    )
}

for problem in all_tests:
    error = do_test(*all_tests[problem])
    print('Error(`{}`) = {}'.format(problem, error))
