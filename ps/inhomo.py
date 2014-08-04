import numpy as np
import math

import matrices
from solver import cart_to_polar

class PsInhomo:

    def extend_inhomo_f(self, nodes=None):
        if nodes is None:
            nodes = self.gamma
    
        ext = np.zeros(len(nodes), dtype=complex)

        for l in range(len(nodes)):
            i, j = nodes[l]
            x, y = self.get_coord(i, j)
            r, th = self.get_polar(i, j)
            etype = self.get_etype(i, j)

            if etype == self.etypes['circle']:
                ext[l] += self.calc_inhomo_circle(r, th)

            elif etype == self.etypes['radius1']:
                ext[l] += self.extend_inhomo_radius(x, y, 1)

            elif etype == self.etypes['radius2']:
                ext[l] += self.extend_inhomo_radius(x, y, 2)

            elif etype == self.etypes['outer1']:
                ext[l] += self.extend_inhomo_outer(x, y, 1)

            elif etype == self.etypes['outer2']:
                ext[l] += self.extend_inhomo_outer(x, y, 2)

        return ext

    def extend_src_f(self):
        R = self.R
        a = self.a

        p = self.problem
        if p.homogeneous:
            return

        for i,j in self.Kplus - self.Mplus:
            x, y = self.get_coord(i, j)
            r, th = self.get_polar(i, j)
            etype = self.get_etype(i, j)

            if etype == self.etypes['circle']:
                x0 = R * np.cos(th)
                y0 = R * np.sin(th)
            elif etype == self.etypes['radius1']:
                x0 = x
                y0 = 0
            elif etype == self.etypes['radius2']:
                x0, y0 = self.get_radius_point(2, x, y)

            elif etype == self.etypes['outer1']:
                x0 = R
                y0 = 0

            elif etype == self.etypes['outer2']:
                x0 = R * np.cos(a)
                y0 = R * np.sin(a)

            vec = np.array((x-x0, y-y0))
            delta = np.linalg.norm(vec)
            vec /= delta

            derivs = [0, 0]

            grad_f = p.eval_grad_f(x0, y0)
            derivs[0] = grad_f.dot(vec)

            hessian_f = p.eval_hessian_f(x0, y0)
            derivs[1] = hessian_f.dot(vec).dot(vec)

            v = p.eval_f(x0, y0)
            for l in range(1, len(derivs)+1):
                v += derivs[l-1] / math.factorial(l) * delta**l

            self.src_f[matrices.get_index(self.N,i,j)] = v

    def inhomo_extend_from_radius(self, Y, f, d_f_Y, d2_f_X, d2_f_Y):
        k = self.problem.k

        derivs = [0, 0, f]
        
        if self.scheme_order == 4:
            derivs.extend([d_f_Y, -d2_f_X - k**2 * f + d2_f_Y])

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * Y**l

        return v

    def get_dir_XY(self, radius_sid):
        if radius_sid == 1:
            dir_X = np.array((1, 0))
            dir_Y = np.array((0, 1))
        elif radius_sid == 2:
            a = self.a
            dir_X = np.array((np.cos(a), np.sin(a)))
            dir_Y = np.array((np.sin(a), -np.cos(a)))

        return dir_X, dir_Y

    def _calc_inhomo_radius(self, x0, y0, dir_X, dir_Y, Y):
        p = self.problem
        if p.homogeneous:
            return 0

        f = p.eval_f(x0, y0)

        if self.scheme_order == 4:
            grad_f = p.eval_grad_f(x0, y0)
            hessian_f = p.eval_hessian_f(x0, y0)
        else:
            grad_f = np.zeros(2)
            hessian_f = np.zeros((2, 2))
        
        d_f_Y = grad_f.dot(dir_Y)

        d2_f_X = hessian_f.dot(dir_X).dot(dir_X)
        d2_f_Y = hessian_f.dot(dir_Y).dot(dir_Y)
        return self.inhomo_extend_from_radius(
            Y, f, d_f_Y, d2_f_X, d2_f_Y)

    def extend_inhomo_radius(self, x, y, radius_sid):
        if self.problem.homogeneous:
            return 0

        if radius_sid == 1:
            x0, y0 = (x, 0)
            Y = y
        elif radius_sid == 2: 
            x0, y0 = self.get_radius_point(radius_sid, x, y) 
            Y = self.signed_dist_to_radius(radius_sid, x, y)

        dir_X, dir_Y = self.get_dir_XY(radius_sid)
        return self._calc_inhomo_radius(x0, y0, dir_X, dir_Y, Y)

    def _extend_inhomo_outer_taylor12(self, x, y, radius_sid):
        p = self.problem
        R = self.R
        a = self.a

        if radius_sid == 1:
            x0, y0 = (R, 0)
            Y = y
            delta = x - x0
        elif radius_sid == 2:
            x0 = R * np.cos(a)
            y0 = R * np.sin(a)

            Y = self.signed_dist_to_radius(radius_sid, x, y)

            x1, y1 = self.get_radius_point(radius_sid, x, y)
            r, th = cart_to_polar(x1, y1)
            delta = r - R

        dir_X, dir_Y = self.get_dir_XY(radius_sid)

        f0 = p.eval_f(x0, y0)
        
        if self.scheme_order == 4:
            grad_f0 = p.eval_grad_f(x0, y0)
            hessian_f0 = p.eval_hessian_f(x0, y0)
        else:
            grad_f0 = np.zeros(2)
            hessian_f0 = np.zeros((2, 2))

        f_derivs = (
            f0, 
            grad_f0.dot(dir_X),
            hessian_f0.dot(dir_X).dot(dir_X)
        )

        f = 0
        for l in range(len(f_derivs)):
            f += f_derivs[l] * delta**l / math.factorial(l)

        d_f_Y_derivs = (
            grad_f0.dot(dir_Y),
            hessian_f0.dot(dir_Y).dot(dir_X),
        )

        d_f_Y = 0
        for l in range(len(d_f_Y_derivs)):
            d_f_Y += d_f_Y_derivs[l] * delta**l / math.factorial(l)

        d2_f_X = hessian_f0.dot(dir_X).dot(dir_X)
        d2_f_Y = hessian_f0.dot(dir_Y).dot(dir_Y)

        return self.inhomo_extend_from_radius(
            Y, f, d_f_Y, d2_f_X, d2_f_Y)

    def _extend_inhomo_outer_taylor0(self, x, y, radius_sid):
        p = self.problem
        R = self.R
        a = self.a

        if radius_sid == 1:
            th0 = 2*np.pi
        elif radius_sid == 2:
            th0 = a

        r, th = cart_to_polar(x, y)
        delta = th - th0

        f0 = p.eval_f_polar(R, th0)
        f_derivs = [f0]
        
        if self.scheme_order == 4:
            d_f_th0 = p.eval_d_f_th(R, th0)
            d2_f_th0 = p.eval_d2_f_th(R, th0)

            f_derivs.extend([d_f_th0, d2_f_th0])

        f = 0
        for l in range(len(f_derivs)):
            f += f_derivs[l] * delta**l / math.factorial(l)

        d_f_r = 0       
        if self.scheme_order == 4:
            d_f_r0 = p.eval_d_f_r(R, th0)
            d2_f_r_th0 = p.eval_d2_f_r_th(R, th0)

            d_f_r_derivs = (d_f_r0, d2_f_r_th0)

            for l in range(len(d_f_r_derivs)):
                d_f_r += d_f_r_derivs[l] * delta**l / math.factorial(l)

            d2_f_r = p.eval_d2_f_r(R, th0)
            d2_f_th = p.eval_d2_f_th(R, th0)
            
        else:
            d2_f_r = 0
            d2_f_th = 0

        return self.extend_inhomo_circle(r, f, d_f_r, d2_f_r, d2_f_th)

    def extend_inhomo_outer(self, x, y, radius_sid):
        if self.problem.homogeneous:
            return 0

        r, th = cart_to_polar(x, y)

        dist0 = r - self.R

        if radius_sid == 1:
            dist1 = y
        elif radius_sid == 2:
            dist1 = self.dist_to_radius(radius_sid, x, y)

        if dist0 < dist1:
            return self._extend_inhomo_outer_taylor0(
                x, y, radius_sid)
        else:
            return self._extend_inhomo_outer_taylor12(
                x, y, radius_sid)
