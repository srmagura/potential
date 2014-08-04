from sympy import *
import numpy as np
import random
import argparse

import problems.jump as jump

def do_test():
    args = symbols('k R r th')
    k, R, r, th = args

    sympy_f_expr = diff(u, r, 2) + diff(u, r)/r + diff(u, th, 2)/r**2 + k**2*u
    print(sympy_f_expr)
    sympy_f_lambda = lambdify(args, sympy_f_expr)

    manual_f_lambda = lambdify(args, manual_f_expr)

    n_trials = 100
    error = np.zeros(n_trials)

    for i in range(n_trials):
        k = random.uniform(.5, 3)
        R = random.uniform(1, 3)
        r = random.uniform(.01, 1)
        th = random.uniform(*th_interval)
    
        sympy_f = sympy_f_lambda(k, R, r, th)
        manual_f = manual_f_lambda(k, R, r, th)
    
        error[i] = abs(sympy_f - manual_f)
    
    print('Error: {}'.format(max(error)))
       
        
choices = ('jump-reg0', 'jump-reg1')

parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True, choices=choices)
cmd_args = parser.parse_args()
problem_name = cmd_args.p

jump_reg_th_interval = (np.pi/6+.01, 2*np.pi-.01)

if problem_name == 'jump-reg0':
    u = -jump.get_u04_expr()
    manual_f_expr = jump.get_reg_f_expr()
    th_interval = jump_reg_th_interval
    
elif problem_name == 'jump-reg1':
    u = -jump.get_u04_sng_expr()
    manual_f_expr = jump.get_reg_f_sng_expr()
    th_interval = jump_reg_th_interval
    
do_test()
