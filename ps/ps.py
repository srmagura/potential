import math
import numpy as np
from numpy import cos, sin

from solver import Solver, Result, cart_to_polar
import matrices

from ps.basis import PsBasis
from ps.grid import PsGrid
from ps.inhomo import PsInhomo
from ps.debug import PsDebug

ETYPE_NAMES = ('standard', 'left', 'right')

TAYLOR_N_DERIVS = 6

class PizzaSolver(Solver, PsBasis, PsGrid, PsInhomo, PsDebug):
    AD_len = 2*np.pi
    R = 2.3

    etypes = dict(zip(ETYPE_NAMES, range(len(ETYPE_NAMES))))

    def __init__(self, problem, N, scheme_order, **kwargs):
        self.a = problem.a
        problem.R = self.R
        
        super().__init__(problem, N, scheme_order, **kwargs)
        
        self.ps_construct_grids()
           
    def get_sid(self, th):
        return self.problem.get_sid(th)

    def is_interior(self, i, j):
        r, th = self.get_polar(i, j)
        return r <= self.R and th >= self.a 

    def get_Q(self, index, ext_only=False):
        columns = []

        for JJ in range(len(self.B_desc)):
            ext = self.extend_basis(JJ, index)
            potential = self.get_potential(ext)
            projection = self.get_trace(potential)

            if not ext_only:
                columns.append(projection - ext)
            else:
                columns.append(ext)
        
        Q = np.column_stack(columns)
                    
        return Q

    def get_radius_point(self, sid, x, y):
        assert sid == 2

        m = np.tan(self.a)
        x1 = (x/m + y)/(m + 1/m)
        y1 = m*x1

        return (x1, y1) 

    def dist_to_radius(self, sid, x, y):
        assert sid == 2
        x1, y1 = self.get_radius_point(sid, x, y)
        return np.sqrt((x1-x)**2 + (y1-y)**2)

    def signed_dist_to_radius(self, sid, x, y):
        unsigned = self.dist_to_radius(sid, x, y)

        m = np.tan(self.a)
        if y > m*x:
            return -unsigned
        else:
            return unsigned

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

    def ext_calc_certain_xi_derivs(self, i, j, param_r, param_th, sid=None):
        xi0 = xi1 = 0
        d2_xi0_arg = d2_xi1_arg = 0
        d4_xi0_arg = 0

        for JJ in range(len(self.B_desc)):
            B = self.eval_dn_B_arg(0, JJ, param_r, param_th, sid)
            d2_B_arg = self.eval_dn_B_arg(2, JJ, param_r, param_th, sid)
            d4_B_arg = self.eval_dn_B_arg(4, JJ, param_r, param_th, sid)

            xi0 += self.c0[JJ] * B
            xi1 += self.c1[JJ] * B

            d2_xi0_arg += self.c0[JJ] * d2_B_arg
            d2_xi1_arg += self.c1[JJ] * d2_B_arg

            d4_xi0_arg += self.c0[JJ] * d4_B_arg

        return (xi0, xi1, d2_xi0_arg, d2_xi1_arg, d4_xi0_arg)
            
    def ext_calc_B_derivs(self, JJ, param_th, segment_sid, index):
        derivs = np.zeros(TAYLOR_N_DERIVS, dtype=complex)

        for n in range(TAYLOR_N_DERIVS):
            derivs[n] = self.eval_dn_B_arg(n, JJ, self.R, param_th, segment_sid)

        return derivs

    def ext_calc_xi_derivs(self, param_th, segment_sid, index): 
        derivs = np.zeros(TAYLOR_N_DERIVS, dtype=complex)

        if index == 0:
            c = self.c0
        elif index == 1:
            c = self.c1

        for n in range(TAYLOR_N_DERIVS):
            for JJ in range(len(self.B_desc)):
                dn_B_arg = self.eval_dn_B_arg(n, JJ, self.R, param_th, segment_sid)
                derivs[n] += c[JJ] * dn_B_arg

        return derivs

    def do_extend_outer(self, i, j, options):
        x, y = self.get_coord(i, j)
        r, th = self.get_polar(i, j)

        taylor_sid = options['taylor_sid']
        delta_arg = options['delta_arg']
        param_th = options['param_th']
        
        if 'JJ' in options:
            JJ = options['JJ']
        else:
            JJ = None
 

        #else:
        #    taylor_sid = radius_sid

        #    if radius_sid == 1:
        #        delta_arg = x - self.R
        #        param_th = 0
        ##        Y = y
        #    elif radius_sid == 2:
        #        x0 = self.R * cos(self.a)
        #        y0 = self.R * sin(self.a)
        #        x1, y1 = self.get_radius_point(radius_sid, x, y)
        #        delta_arg = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        #        param_th = self.a
        #        Y = dist1

        derivs0 = np.zeros(TAYLOR_N_DERIVS)
        derivs1 = np.zeros(TAYLOR_N_DERIVS)

        if JJ is None:
            derivs0 = self.ext_calc_xi_derivs(param_th, taylor_sid, 0) 
            derivs1 = self.ext_calc_xi_derivs(param_th, taylor_sid, 1) 
        elif index == 0:
            derivs0 = self.ext_calc_B_derivs(JJ, param_th, taylor_sid, 0) 
        elif index == 1:
            derivs1 = self.ext_calc_B_derivs(JJ, param_th, taylor_sid, 1) 

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
            v = self.extend_circle(r, *ext_params)
        elif taylor_sid in {1, 2}:
            v = self.extend_from_radius(Y, *ext_params)

        return v

    def do_extend_0_standard(self, i, j):
        r, th = self.get_polar(i, j)       
        derivs = self.ext_calc_certain_xi_derivs(i, j, self.R, th, sid=0)            
        return self.extend_circle(r, *derivs)
        
    def do_extend_0_left(self, i, j):
        x, y = self.get_coord(i, j)
        r, th = self.get_polar(i, j)

        options = {
            'taylor_sid': 0,
            'delta_arg': th - self.a,
            'param_th': self.a
        }
         
        return self.do_extend_outer(i, j, options)
        
    def do_extend_0_right(self, i, j):
        x, y = self.get_coord(i, j)
        r, th = self.get_polar(i, j)

        options = {
            'taylor_sid': 0,
            'delta_arg': th,
            'param_th': 2*np.pi
        }
         
        return self.do_extend_outer(i, j, options)
        
    def do_extend_1_standard(self, i, j):
        x, y = self.get_coord(i, j)
        derivs = self.ext_calc_certain_xi_derivs(i, j, x, 2*np.pi)

        return self.extend_from_radius(y, *derivs)

    def do_extend_2_standard(self, i, j):
        x, y = self.get_coord(i, j)
        x0, y0 = self.get_radius_point(2, x, y)

        param_r = cart_to_polar(x0, y0)[0]
        derivs = self.ext_calc_certain_xi_derivs(i, j, param_r, self.a, sid=2)

        Y = self.signed_dist_to_radius(2, x, y)
        return self.extend_from_radius(Y, *derivs)

    def extend_boundary(self):
        R = self.R
        k = self.problem.k

        all_ext = {} 
        
        for sid in range(3):
            gamma = self.all_gamma[sid]
            
            ext = np.zeros(len(gamma), dtype=complex)
            all_ext[sid] = ext

            for l in range(len(gamma)):
                i, j = gamma[l]
                #x, y = self.get_coord(i, j)
                #r, th = self.get_polar(i, j)
                etype = self.get_etype(sid, i, j)

                if sid == 0:
                    if etype == self.etypes['standard']:
                        ext[l] = self.do_extend_0_standard(i, j)
                    elif etype == self.etypes['left']:
                        ext[l] = self.do_extend_0_left(i, j)
                    elif etype == self.etypes['right']:
                        ext[l] = self.do_extend_0_right(i, j)
                        
                elif sid == 1:
                    if etype == self.etypes['standard']:
                        ext[l] = self.do_extend_1_standard(i, j)
                        
                elif sid == 2:
                    if etype == self.etypes['standard']:
                        ext[l] = self.do_extend_2_standard(i, j)
                        
        #boundary += self.extend_inhomo_f(nodes)
        return all_ext
        
    def run(self):
        n_basis_tuple = self.problem.get_n_basis(self.N)
        #print('n_basis_tuple: {}'.format(n_basis_tuple))
        self.setup_B_desc(*n_basis_tuple)
        
        return self.test_extend_boundary({
            (0, 'standard'),
            (0, 'left'),
            (0, 'right'),
            (1, 'standard'),
            (2, 'standard'),
        })
        #return self.test_extend_basis()

        #self.calc_c0()
        #self.c0_test()
        #self.calc_c1()
        #self.c1_test()
        #self.print_c1()
        #self.test_extend_basis_not()
        return self.plot_gamma()
        #self.test_with_c1_exact()

        #ext = self.extend_boundary()
        #u_act = self.get_potential(ext) + self.ap_sol_f
        
        #self.plot_contour(u_act)

        #error = self.eval_error(u_act)
        
        result = Result()
        result.error = error
        result.u_act = u_act
        return result
