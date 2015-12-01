import numpy as np
import math

import matrices
from solver import cart_to_polar
from .multivalue import Multivalue

class PsInhomo:

    def _extend_f_get_sid(self, i, j):
        R = self.R
        a = self.a

        r, th = self.get_polar(i, j)

        if self.get_etype(0, i, j) == self.etypes['standard']:
            return 0
        elif r < self.R:
            if th < a/2:
                return 1
            else:
                return 2
        else:
            return 0

    def extend_f(self):
        R = self.R
        a = self.a

        p = self.problem
        if p.homogeneous:
            return

        for i,j in self.Kplus - self.global_Mplus:
            x, y = self.get_coord(i, j)
            r, th = self.get_polar(i, j)

            sid = self._extend_f_get_sid(i, j)
            etype = self.get_etype(sid, i, j)

            if sid == 1:
                x0 = x
                y0 = 0
            elif sid == 2:
                x0, y0 = self.get_radius_point(2, x, y)
            elif sid == 0:
                if etype == self.etypes['standard']:
                    x0 = R * np.cos(th)
                    y0 = R * np.sin(th)
                elif etype == self.etypes['left']:
                    x0 = R * np.cos(a)
                    y0 = R * np.sin(a)
                elif etype == self.etypes['right']:
                    x0 = R
                    y0 = 0

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

            self.f[matrices.get_index(self.N,i,j)] = v

    def inhomo_extend_from_radius(self, Y, f, d_f_Y, d2_f_X, d2_f_Y):
        k = self.problem.k

        derivs = [0, 0, f]

        if self.extension_order > 2:
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

    def _extend_inhomo_12_outer(self, i, j, radius_sid):
        p = self.problem
        R = self.R
        a = self.a

        x, y = self.get_coord(i, j)

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

        if self.extension_order > 2:
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

        return {
            'elen': delta + abs(Y),
            'value': self.inhomo_extend_from_radius(
                Y, f, d_f_Y, d2_f_X, d2_f_Y)
        }

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

    def do_extend_inhomo_0_standard(self, i, j):
        r, th = self.get_polar(i, j)
        return {
            'elen': abs(self.R - r),
            'value': self.calc_inhomo_circle(r, th),
        }

    def _extend_inhomo_0_outer(self, i, j, radius_sid):
        p = self.problem
        R = self.R
        a = self.a

        x, y = self.get_coord(i, j)
        r, th = self.get_polar(i, j)

        if radius_sid == 1:
            delta = th
            th0 = 2*np.pi
        elif radius_sid == 2:
            th0 = a
            delta = th - th0

        f0 = p.eval_f_polar(R, th0)
        f_derivs = [f0]

        if self.extension_order > 3:
            d_f_th0 = p.eval_d_f_th(R, th0)
            d2_f_th0 = p.eval_d2_f_th(R, th0)

            f_derivs.extend([d_f_th0, d2_f_th0])

        f = 0
        for l in range(len(f_derivs)):
            f += f_derivs[l] * delta**l / math.factorial(l)

        d_f_r = 0
        if self.extension_order > 3:
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

        return {
            'elen': self.R*abs(delta) + abs(self.R - r),
            'value': self.extend_inhomo_circle(r, f, d_f_r, d2_f_r, d2_f_th)
        }

    def do_extend_inhomo_0_left(self, i, j):
        return self._extend_inhomo_0_outer(i, j, 2)

    def do_extend_inhomo_0_right(self, i, j):
        return self._extend_inhomo_0_outer(i, j, 1)

    def _calc_inhomo_radius(self, x0, y0, dir_X, dir_Y, Y):
        p = self.problem
        f = p.eval_f(x0, y0)

        if self.extension_order > 3 and Y != 0:
            if x0 == 0 and y0 == 0:
                '''
                Use Taylor's theorem to construct a smooth extension of
                f (?), grad_f, and hessian_f at the origin, since they may
                be undefined or annoying to calculate at this point.
                '''
                h = self.AD_len / (10*self.N)

                hessian_f = p.eval_hessian_f(-h, 0)

                _grad_f = p.eval_grad_f(-h, 0)
                grad_f = _grad_f + h * hessian_f.dot((1, 0))

            else:
                grad_f = p.eval_grad_f(x0, y0)
                hessian_f = p.eval_hessian_f(x0, y0)
        else:
            grad_f = np.zeros(2)
            hessian_f = np.zeros((2, 2))

        d_f_Y = grad_f.dot(dir_Y)

        d2_f_X = hessian_f.dot(dir_X).dot(dir_X)
        d2_f_Y = hessian_f.dot(dir_Y).dot(dir_Y)

        elen = abs(Y)

        # Only so that elen matches that of the homogeneous extension
        r, th = cart_to_polar(x0, y0)
        if x0 < 0 or r > self.R:
            elen += r

        return {
            'elen': elen,
            'value':self.inhomo_extend_from_radius(
                Y, f, d_f_Y, d2_f_X, d2_f_Y)
        }

    def _extend_inhomo_radius(self, i, j, radius_sid):
        x, y = self.get_coord(i, j)

        if radius_sid == 1:
            x0, y0 = (x, 0)
            Y = y
        elif radius_sid == 2:
            x0, y0 = self.get_radius_point(radius_sid, x, y)
            Y = self.signed_dist_to_radius(radius_sid, x, y)

        dir_X, dir_Y = self.get_dir_XY(radius_sid)
        return self._calc_inhomo_radius(x0, y0, dir_X, dir_Y, Y)

    def do_extend_inhomo_1_standard(self, i, j):
        return self._extend_inhomo_radius(i, j, 1)

    def do_extend_inhomo_1_left(self, i, j):
        return self._extend_inhomo_radius(i, j, 1)

    def do_extend_inhomo_1_right(self, i, j):
        return self._extend_inhomo_12_outer(i, j, 1)

    def do_extend_inhomo_2_standard(self, i, j):
        return self._extend_inhomo_radius(i, j, 2)

    def do_extend_inhomo_2_left(self, i, j):
        return self._extend_inhomo_radius(i, j, 2)

    def do_extend_inhomo_2_right(self, i, j):
        return self._extend_inhomo_12_outer(i, j, 2)

    def mv_extend_inhomo_f(self):
        R = self.R
        k = self.problem.k

        mv_ext = Multivalue(self.union_gamma)

        for sid in range(self.N_SEGMENT):
            gamma = self.all_gamma[sid]

            for (i, j) in gamma:
                etype = self.get_etype(sid, i, j)
                result = None

                if self.problem.homogeneous:
                    result = {'elen': 0, 'value': 0}

                elif sid == 0:
                    if etype == self.etypes['standard']:
                        result = self.do_extend_inhomo_0_standard(i, j)
                    elif etype == self.etypes['left']:
                        result = self.do_extend_inhomo_0_left(i, j)
                    elif etype == self.etypes['right']:
                        result = self.do_extend_inhomo_0_right(i, j)

                elif sid == 1:
                    if etype == self.etypes['standard']:
                        result = self.do_extend_inhomo_1_standard(i, j)
                    elif etype == self.etypes['left']:
                        result = self.do_extend_inhomo_1_left(i, j)
                    elif etype == self.etypes['right']:
                        result = self.do_extend_inhomo_1_right(i, j)

                elif sid == 2:
                    if etype == self.etypes['standard']:
                        result = self.do_extend_inhomo_2_standard(i, j)
                    elif etype == self.etypes['left']:
                        result = self.do_extend_inhomo_2_left(i, j)
                    elif etype == self.etypes['right']:
                        result = self.do_extend_inhomo_2_right(i, j)

                result['setype'] = (sid, etype)
                mv_ext[(i, j)].append(result)

        return mv_ext

    def extend_inhomo_f(self):
        return self.mv_extend_inhomo_f().reduce()
