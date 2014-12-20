from sympy import *
import numpy as np

from solver import cart_to_polar   
   
class SympyProblem:
    '''
    Use SymPy to symbolically differentiate the source function f.
    Calculates up to 2nd order derivatives of f w.r.t. r and th, 
    as well as the gradient and Hessian of f.
    '''

    def __init__(self, **kwargs):
        '''
        Requires a keyword argument f which is a SymPy expression
        in k, R, r, and th.
        '''
        f = kwargs.pop('f_expr')
        #super().__init__(**kwargs)
          
        args = symbols('k R r th')
        k, R, r, th = args
        
        self.f_polar_lambda = lambdify(args, f)
        
        # If using 2nd order scheme, don't need derivatives of f
        if kwargs['scheme_order'] == 4:
            self.do_diff(f)
        
    def do_diff(self, f):
        '''
        Do the differentiation.
        '''
        args = symbols('k R r th')
        k, R, r, th = args
    
        d_f_r = diff(f, r)
        self.d_f_r_lambda = lambdify(args, d_f_r)
        
        d2_f_r = diff(f, r, 2)
        self.d2_f_r_lambda = lambdify(args, d2_f_r)
        
        d_f_th = diff(f, th)
        self.d_f_th_lambda = lambdify(args, d_f_th)
        
        d2_f_th = diff(f, th, 2)
        self.d2_f_th_lambda = lambdify(args, d2_f_th)
        
        d2_f_r_th = diff(f, r, th)
        self.d2_f_r_th_lambda = lambdify(args, d2_f_r_th)
        
        x, y = symbols('x y')
        cart_args = (k, R, x, y)
        
        subs_dict_upper = {
            r: sqrt(x**2 + y**2),
            th: atan2(y, x)
        }
        
        subs_dict_lower = {
            r: sqrt(x**2 + y**2),
            th: atan2(y, x) + 2*pi
        }
        
        f_cart_upper = f.subs(subs_dict_upper)
        f_cart_lower = f.subs(subs_dict_lower)
        
        # Hessian
        d2_f_x_upper = diff(f_cart_upper, x, 2)
        self.d2_f_x_upper_lambda = lambdify(cart_args, d2_f_x_upper)
        
        d2_f_x_lower = diff(f_cart_lower, x, 2)
        self.d2_f_x_lower_lambda = lambdify(cart_args, d2_f_x_lower)
        
        d2_f_x_y_upper = diff(f_cart_upper, x, y)
        self.d2_f_x_y_upper_lambda = lambdify(cart_args, d2_f_x_y_upper)
        
        d2_f_x_y_lower = diff(f_cart_lower, x, y)
        self.d2_f_x_y_lower_lambda = lambdify(cart_args, d2_f_x_y_lower)
        
        d2_f_y_upper = diff(f_cart_upper, y, 2)
        self.d2_f_y_upper_lambda = lambdify(cart_args, d2_f_y_upper)
        
        d2_f_y_lower = diff(f_cart_lower, y, 2)
        self.d2_f_y_lower_lambda = lambdify(cart_args, d2_f_y_lower)

    def eval_f_polar(self, r, th):
        return self.f_polar_lambda(self.k, self.R, r, th)
    
    def eval_d_f_r(self, r, th):
        return self.d_f_r_lambda(self.k, self.R, r, th)

    def eval_d2_f_r(self, r, th):
        return self.d2_f_r_lambda(self.k, self.R, r, th)

    def eval_d_f_th(self, r, th):
        return self.d_f_th_lambda(self.k, self.R, r, th)

    def eval_d2_f_th(self, r, th):   
        return self.d2_f_th_lambda(self.k, self.R, r, th)

    def eval_d2_f_r_th(self, r, th):  
        return self.d2_f_r_th_lambda(self.k, self.R, r, th)

    def eval_grad_f(self, x, y):        
        r, th = cart_to_polar(x, y)
        d_f_r = self.eval_d_f_r(r, th)
        d_f_th = self.eval_d_f_th(r, th)
        
        d_f_x = d_f_r * np.cos(th) - d_f_th * np.sin(th) / r
        d_f_y = d_f_r * np.sin(th) + d_f_th * np.cos(th) / r
        
        return np.array((d_f_x, d_f_y))

    def eval_hessian_f(self, x, y): 
        if y > 0:
            d2_f_x = self.d2_f_x_upper_lambda(self.k, self.R, x, y)
            d2_f_x_y = self.d2_f_x_y_upper_lambda(self.k, self.R, x, y)
            d2_f_y = self.d2_f_y_upper_lambda(self.k, self.R, x, y)
        else:
            d2_f_x = self.d2_f_x_lower_lambda(self.k, self.R, x, y)
            d2_f_x_y = self.d2_f_x_y_lower_lambda(self.k, self.R, x, y)
            d2_f_y = self.d2_f_y_lower_lambda(self.k, self.R, x, y)
        
        return np.array(((d2_f_x, d2_f_x_y), (d2_f_x_y, d2_f_y)))
	
