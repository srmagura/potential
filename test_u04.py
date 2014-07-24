import random
import numpy as np

from solver import cart_to_polar

from problems.jump import eval_u04 as eval_u04_raw, eval_reg_f as eval_reg_f_raw


def eval_u04(x, y):
    r, th = cart_to_polar(x, y)
    return eval_u04_raw(k, R, r, th)
    
def eval_reg_f(x, y):
    r, th = cart_to_polar(x, y)
    return eval_reg_f_raw(k, R, r, th)
    
def eval_lhs(x0, y0):
    d2x = (eval_u04(x0+h, y0) - 2*eval_u04(x0, y0) + eval_u04(x0-h, y0))/h**2
    d2y = (eval_u04(x0, y0+h) - 2*eval_u04(x0, y0) + eval_u04(x0, y0-h))/h**2

    return -(d2x + d2y + k**2*eval_u04(x0, y0))

def do_test(h):
    error = []

    for i in range(250):
        x0 = random.uniform(-1, -.1)
        y0 = random.uniform(-1, 1)

        lhs = eval_lhs(x0, y0)
        rhs = eval_reg_f(x0, y0)
        error.append(abs(lhs-rhs))

    return max(error)

k = 1
R = 1

p = -4
while p >= -10:
    h = 2**p
    p -= 1
    
    error = do_test(h)
    print('h={}    error={}'.format(h, error))

