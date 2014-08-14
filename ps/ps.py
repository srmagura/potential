import math
import numpy as np
from numpy import cos, sin

from solver import Solver, Result, cart_to_polar
import matrices

from ps.basis import PsBasis
from ps.grid import PsGrid
from ps.extend import PsExtend
from ps.inhomo import PsInhomo
from ps.debug import PsDebug


class PizzaSolver(Solver, PsBasis, PsGrid, PsExtend, PsInhomo, PsDebug):
    AD_len = 2*np.pi
    R = 2.3

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
        
    def get_w2(self, all_ext):
        w2 = np.zeros([(self.N-1)**2], dtype=complex)
        test_w2_set = {}
               
        for sid in range(3):
            gamma = self.all_gamma[sid]
            ext = all_ext[sid]

            for l in range(len(gamma)):
                i, j = gamma[l]
                index = matrices.get_index(self.N, i, j) 
                
                do_set = False
                
                in_gamma1 = (i, j) in self.all_gamma[1]
                in_gamma2 = (i, j) in self.all_gamma[2]
                
                in_Mplus1 = (i, j) in self.all_Mplus[1]
                in_Mplus2 = (i, j) in self.all_Mplus[2]
                
                if sid == 0:
                    do_set = True
                elif in_gamma1 and in_gamma2:
                    if (i, j) in self.all_Mplus[sid]:
                        do_set = True
                    elif not (in_Mplus1 or in_Mplus2):
                        do_set = True
                else:
                    do_set = True
                
                if do_set:
                    w2[index] = ext[l]
                    test_w2_set[index] = True
       
        error = []
        for gamma in self.all_gamma.values():
            for i, j in gamma:
                index = matrices.get_index(self.N, i, j)
                x, y = self.get_coord(i, j)
                assert index in test_w2_set
                
                error.append(abs(w2[index]-self.problem.eval_expected(x, y)))
        
        print('w2 error: {}'.format(max(error)))
        return w2
        
        
    def get_potential(self, all_ext):
        union_w = np.zeros((self.N-1)**2, dtype=complex)
        for sid in range(3):                    
            gamma = self.all_gamma[sid]
            ext = all_ext[sid]
                      
            for l in range(len(gamma)):
                i, j = gamma[l]
                index = matrices.get_index(self.N, i, j) 
                union_w[index] = ext[l]

        union_Lw = np.ravel(self.L.dot(union_w))
        radius_Lw = {}
        
        for rsid in (1, 2):
            w = np.zeros((self.N-1)**2, dtype=complex)
            
            gamma = self.all_gamma[rsid]
            ext = all_ext[rsid]
            
            for l in range(len(gamma)):
                i, j = gamma[l]
                index = matrices.get_index(self.N, i, j) 
                w[index] = ext[l]
                
            radius_Lw[rsid] = np.ravel(self.L.dot(w))
            
        rhs = np.zeros((self.N-1)**2, dtype=complex)
        
        for i, j in self.global_Mplus:
            index = matrices.get_index(self.N, i, j)
            r, th = self.get_coord(i, j)
            
            if r < self.R/2:
                in_Mplus1 = (i, j) in self.all_Mplus[1]
                in_Mplus2 = (i, j) in self.all_Mplus[2]
            
                if in_Mplus1 and not in_Mplus2:
                    rhs[index] = radius_Lw[1][index]
                elif not in_Mplus1 and in_Mplus2:
                    rhs[index] = radius_Lw[2][index]
                elif not in_Mplus1 and not in_Mplus2:
                    rhs[index] = union_Lw[index]
                else:
                    raise Exception('{}'.format((i, j)))
            else:
                rhs[index] = union_Lw[index]

        w2 = self.get_w2(all_ext)                           
        return w2 - self.LU_factorization.solve(rhs)

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
        
    def run(self):
        n_basis_tuple = self.problem.get_n_basis(self.N)
        #print('n_basis_tuple: {}'.format(n_basis_tuple))
        self.setup_B_desc(*n_basis_tuple)
        
        #return self.test_extend_boundary()
        #return self.test_extend_boundary({
        #    (0, 'standard'),
        #    (0, 'left'),
        #    (0, 'right'),
        #    (1, 'standard'),
        #    (1, 'left'),
        #    (1, 'right'),
        #    (2, 'standard'),
        #    (2, 'left'),
        #    (2, 'right'),
        #})
        #return self.test_extend_basis()

        self.calc_c0()
        self.calc_c1_exact()
        #self.c0_test()
        #self.calc_c1()
        #self.c1_test()
        #self.print_c1()
        #self.test_extend_basis_not()
        self.plot_gamma()
        #self.test_with_c1_exact()

        all_ext = self.extend_boundary()
        u_act = self.get_potential(all_ext) #+ self.ap_sol_f
        
        #self.plot_contour(u_act)

        error = self.eval_error(u_act)
        
        result = Result()
        result.error = error
        result.u_act = u_act
        return result
