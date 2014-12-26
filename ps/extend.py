import numpy as np
from numpy import sin, cos
import math

from solver import cart_to_polar

ETYPE_NAMES = ('standard', 'left', 'right')
TAYLOR_N_DERIVS = 6

class PsExtend:

    etypes = dict(zip(ETYPE_NAMES, range(len(ETYPE_NAMES))))

    def get_etype(self, sid, i, j):
        a = self.a
        R = self.R
    
        x, y = self.get_coord(i, j)
        r, th = self.get_polar(i, j)
        
        if sid == 0:
            if th < a/2:
                return self.etypes['right']
            elif th < a:
                return self.etypes['left']
            else:
                return self.etypes['standard']
                
        elif sid == 1:
            if x < 0:
                return self.etypes['left']
            elif x > R:
                return self.etypes['right']
            else:
                return self.etypes['standard']
                
        elif sid == 2:
            slope = -1/np.tan(a)
            if y < slope*x:
                return self.etypes['left']
            elif y - R*np.sin(a) > slope*(x - R*np.cos(a)):
                return self.etypes['right']
            else:
                return self.etypes['standard']      

    def extend_from_radius(self, Y, xi0, xi1, d2_xi0_X,
        d2_xi1_X, d4_xi0_X):

        k = self.k

        derivs = []
        derivs.append(xi0)
        derivs.append(xi1)
        derivs.append(-(d2_xi0_X + k**2 * xi0))

        derivs.append(-(d2_xi1_X + k**2 * xi1))

        derivs.append(d4_xi0_X + k**2 * (2*d2_xi0_X + k**2 * xi0))

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * Y**l

        return v

    def ext_calc_certain_xi_derivs(self, i, j, param_r, param_th, options):
        if 'sid' in options:
            sid = options['sid']
        else:
            sid = None
            
        if 'JJ' in options:
            JJ_list = (options['JJ'],)
            c0 = c1 = {options['JJ']: 1}
            index = options['index']
        else:
            JJ_list = range(len(self.B_desc))
            c0 = self.c0
            c1 = self.c1
            index = None
               
        xi0 = xi1 = 0
        d2_xi0_arg = d2_xi1_arg = 0
        d4_xi0_arg = 0

        for JJ in JJ_list:
            B = self.eval_dn_B_arg(0, JJ, param_r, param_th, sid)
            d2_B_arg = self.eval_dn_B_arg(2, JJ, param_r, param_th, sid)
            d4_B_arg = self.eval_dn_B_arg(4, JJ, param_r, param_th, sid)

            if index is None or index == 0:
                xi0 += c0[JJ] * B
                d2_xi0_arg += c0[JJ] * d2_B_arg
                d4_xi0_arg += c0[JJ] * d4_B_arg
                
            if index is None or index == 1:
                xi1 += c1[JJ] * B
                d2_xi1_arg += c1[JJ] * d2_B_arg

        return (xi0, xi1, d2_xi0_arg, d2_xi1_arg, d4_xi0_arg)
            
    def ext_calc_B_derivs(self, JJ, sid, param_r, param_th):
        derivs = np.zeros(TAYLOR_N_DERIVS, dtype=complex)

        for n in range(TAYLOR_N_DERIVS):
            derivs[n] = self.eval_dn_B_arg(n, JJ, param_r, param_th, sid)

        return derivs

    def ext_calc_xi_derivs(self, param_r, param_th, sid, index): 
        derivs = np.zeros(TAYLOR_N_DERIVS, dtype=complex)

        if index == 0:
            c = self.c0
        elif index == 1:
            c = self.c1

        for n in range(TAYLOR_N_DERIVS):
            for JJ in range(len(self.B_desc)):
                dn_B_arg = self.eval_dn_B_arg(n, JJ, param_r, param_th, sid)
                derivs[n] += c[JJ] * dn_B_arg

        return derivs
        
    def do_extend_taylor(self, i, j, options):
        taylor_sid = options['taylor_sid']
        delta_arg = options['delta_arg']
        
        param_r = options['param_r']
        param_th = options['param_th']
        
        if 'JJ' in options:
            JJ = options['JJ']
            index = options['index']
        else:
            JJ = None
            index = None

        derivs0 = np.zeros(TAYLOR_N_DERIVS)
        derivs1 = np.zeros(TAYLOR_N_DERIVS)

        if JJ is None:
            derivs0 = self.ext_calc_xi_derivs(param_r, param_th, taylor_sid, 0) 
            derivs1 = self.ext_calc_xi_derivs(param_r, param_th, taylor_sid, 1) 
        elif index == 0:
            derivs0 = self.ext_calc_B_derivs(JJ, taylor_sid, param_r, param_th) 
        elif index == 1:
            derivs1 = self.ext_calc_B_derivs(JJ, taylor_sid, param_r, param_th)

        xi0 = xi1 = 0
        d2_xi0_arg = d2_xi1_arg = 0
        d4_xi0_arg = 0

        for l in range(len(derivs0)):
            fac = delta_arg**l / math.factorial(l)
            xi0 += derivs0[l] * fac 
            xi1 += derivs1[l] * fac

            ll = l - 2
            if ll >= 0:
                fac = delta_arg**ll / math.factorial(ll)
                d2_xi0_arg += derivs0[l] * fac
                d2_xi1_arg += derivs1[l] * fac

            ll = l - 4
            if ll >= 0:
                fac = delta_arg**ll / math.factorial(ll)
                d4_xi0_arg += derivs0[l] * fac

        ext_params = (xi0, xi1, d2_xi0_arg, d2_xi1_arg, d4_xi0_arg)

        if taylor_sid == 0:
            r, th = self.get_polar(i, j)
            v = self.extend_circle(r, *ext_params)
            
            elen = delta_arg*self.R + r
            
        elif taylor_sid in {1, 2}:
            Y = options['Y']
            v = self.extend_from_radius(Y, *ext_params)
            
            elen = delta_arg + abs(Y)

        return {'elen': elen, 'value': v}
        
    def do_extend_12_left(self, i, j, options):
        options['param_r'] = 0
        return self.do_extend_taylor(i, j, options)

    def do_extend_12_right(self, i, j, options):
        options['param_r'] = self.R
        return self.do_extend_taylor(i, j, options)

    def do_extend_0_standard(self, i, j, options):
        options['sid'] = 0
        r, th = self.get_polar(i, j)       
        derivs = self.ext_calc_certain_xi_derivs(i, j, self.R, th, options)            
        return {'elen': abs(self.R - r), 'value': self.extend_circle(r, *derivs)}
        
    def do_extend_0_left(self, i, j, options):
        x, y = self.get_coord(i, j)
        r, th = self.get_polar(i, j)

        options.update({
            'taylor_sid': 0,
            'delta_arg': th - self.a,
            'param_th': self.a
        })
         
        return self.do_extend_12_right(i, j, options)
        
    def do_extend_0_right(self, i, j, options):
        r, th = self.get_polar(i, j)

        options.update({
            'taylor_sid': 0,
            'delta_arg': th,
            'param_th': 2*np.pi
        })
         
        return self.do_extend_12_right(i, j, options)
          
    def do_extend_1_standard(self, i, j, options):
        x, y = self.get_coord(i, j)
        options['sid'] = 1
        derivs = self.ext_calc_certain_xi_derivs(i, j, x, 2*np.pi, options)

        return {'elen': abs(y), 'value': self.extend_from_radius(y, *derivs)}
                
    def do_extend_1_left(self, i, j, options):
        x, y = self.get_coord(i, j)
        r, th = self.get_polar(i, j)

        options.update({
            'taylor_sid': 1,
            'delta_arg': x,
            'param_th': 2*np.pi,
            'Y': y,
        })
         
        return self.do_extend_12_left(i, j, options)      
        
    def do_extend_1_right(self, i, j, options):
        x, y = self.get_coord(i, j)

        options.update({
            'taylor_sid': 1,
            'delta_arg': x - self.R,
            'param_th': 2*np.pi,
            'Y': y
        })
         
        return self.do_extend_12_right(i, j, options)

    def do_extend_2_standard(self, i, j, options):
        x, y = self.get_coord(i, j)
        x0, y0 = self.get_radius_point(2, x, y)

        param_r = cart_to_polar(x0, y0)[0]
        options['sid'] = 2
        derivs = self.ext_calc_certain_xi_derivs(i, j, param_r, self.a, options)

        Y = self.signed_dist_to_radius(2, x, y)
        return {'elen': abs(Y), 'value': self.extend_from_radius(Y, *derivs)}
        
    def do_extend_2_left(self, i, j, options):
        x, y = self.get_coord(i, j)      
        x1, y1 = self.get_radius_point(2, x, y)

        options.update({
            'taylor_sid': 2,
            'delta_arg': -np.sqrt(x1**2 + y1**2),
            'param_th': 2*np.pi,
            'Y': self.signed_dist_to_radius(2, x, y)
        })
         
        return self.do_extend_12_left(i, j, options)
        
    def do_extend_2_right(self, i, j, options):
        R = self.R
        a = self.a
    
        x, y = self.get_coord(i, j)
        
        x0 = R*cos(a)
        y0 = R*sin(a)
        x1, y1 = self.get_radius_point(2, x, y)

        options.update({
            'taylor_sid': 2,
            'delta_arg': np.sqrt((x1 - x0)**2 + (y1 - y0)**2),
            'param_th': a,
            'Y': self.signed_dist_to_radius(2, x, y)
        })
         
        return self.do_extend_12_right(i, j, options)

    def reduce_all_ext(self, all_ext):
        ext = np.zeros(len(self.union_gamma), dtype=complex)
        
        for l in range(len(self.union_gamma)):
            i, j = self.union_gamma[l]
            ext_list = all_ext[(i, j)]
            
            if len(ext_list) == 1:
                ext[l] = ext_list[0]['value']
            elif len(ext_list) == 2:                
                elen0 = ext_list[0]['elen']
                value0 = ext_list[0]['value']
                
                elen1 = ext_list[1]['elen']
                value1 = ext_list[1]['value']
                
                if elen0 == 0 and elen1 == 0:
                    elen0 = elen1 = 1
                
                ext[l] = (elen1*value0 + elen0*value1) / (elen0 + elen1)
                
            else:
                assert False
                
        return ext
            

    def extend_boundary(self, options={}):
        R = self.R
        k = self.problem.k

        all_ext = {}
        
        for sid in range(3):
            gamma = self.all_gamma[sid]         

            for l in range(len(gamma)):
                i, j = gamma[l]
                etype = self.get_etype(sid, i, j)
            
                if (i, j) not in all_ext:
                    all_ext[(i, j)] = []
                    
                ext_list = all_ext[(i, j)]
            
                if sid == 0:
                    if etype == self.etypes['standard']:
                        ext_list.append(self.do_extend_0_standard(i, j, options))
                    elif etype == self.etypes['left']:
                        ext_list.append(self.do_extend_0_left(i, j, options))
                    elif etype == self.etypes['right']:
                        ext_list.append(self.do_extend_0_right(i, j, options))
                        
                elif sid == 1:
                    if etype == self.etypes['standard']:
                        ext_list.append(self.do_extend_1_standard(i, j, options))
                    elif etype == self.etypes['left']:
                        ext_list.append(self.do_extend_1_left(i, j, options))
                    elif etype == self.etypes['right']:
                        ext_list.append(self.do_extend_1_right(i, j, options))
                        
                elif sid == 2:
                    if etype == self.etypes['standard']:
                        ext_list.append(self.do_extend_2_standard(i, j, options))
                    elif etype == self.etypes['left']:
                        ext_list.append(self.do_extend_2_left(i, j, options))
                    elif etype == self.etypes['right']:
                        ext_list.append(self.do_extend_2_right(i, j, options))

        ext = self.reduce_all_ext(all_ext)
        
        if 'JJ' in options:
            # Don't add inhomogeneous part if this is a basis function
            return ext
        else:
            return ext + self.extend_inhomo_f()
