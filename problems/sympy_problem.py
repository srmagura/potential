from sympy import *
import numpy as np
   
   
class SympyProblem:

    def __init__(self, **kwargs):
        f = kwargs.pop('f_expr')
        #super().__init__(**kwargs)
          
        args = symbols('k R r th')
        k, R, r, th = args
        
        self.f_polar_lambda = lambdify(args, f)
        
        if kwargs['scheme_order'] == 4:
            self.do_diff(f)
        
    def do_diff(self, f):
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
        
        # Gradient
        d_f_x_upper = diff(f_cart_upper, x)
        self.d_f_x_upper_lambda = lambdify(cart_args, d_f_x_upper)
        
        d_f_x_lower = diff(f_cart_lower, x)
        self.d_f_x_lower_lambda = lambdify(cart_args, d_f_x_lower)
        
        d_f_y_upper = diff(f_cart_upper, y)
        self.d_f_y_upper_lambda = lambdify(cart_args, d_f_y_upper)
        
        d_f_y_lower = diff(f_cart_lower, y)
        self.d_f_y_lower_lambda = lambdify(cart_args, d_f_y_lower)
        
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
        assert th > .01
        return self.f_polar_lambda(self.k, self.R, r, th)
    
    def eval_d_f_r(self, r, th):
        assert th > .01
        return self.d_f_r_lambda(self.k, self.R, r, th)

    def eval_d2_f_r(self, r, th):
        assert th > .01
        return self.d2_f_r_lambda(self.k, self.R, r, th)

    def eval_d_f_th(self, r, th):
        assert th > .01
        return self.d_f_th_lambda(self.k, self.R, r, th)

    def eval_d2_f_th(self, r, th):   
        assert th > .01
        return self.d2_f_th_lambda(self.k, self.R, r, th)

    def eval_d2_f_r_th(self, r, th):  
        assert th > .01
        return self.d2_f_r_th_lambda(self.k, self.R, r, th)

    def eval_grad_f(self, x, y):
        if y > 0:
            d_f_x = self.d_f_x_upper_lambda(self.k, self.R, x, y)
            d_f_y = self.d_f_y_upper_lambda(self.k, self.R, x, y)
        else:
            #print('-------- x={}     y={} --------------'.format(x,y))
            d_f_x = self.d_f_x_lower_lambda(self.k, self.R, x, y)
            d_f_y = self.d_f_y_lower_lambda(self.k, self.R, x, y)
        
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
