import numpy as np
import math

import matrices

class PsInhomo:

    def extend_inhomogeneous_f(self, *args):
        ext = np.zeros(len(self.gamma), dtype=complex)
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

    def _extend_inhomogeneous_radius(self, x0, y0, dir_X, dir_Y, Y):
        p = self.problem
        if p.homogeneous:
            return 0

        k = p.k

        derivs = [0, 0]
        derivs.append(p.eval_f(x0, y0))

        grad_f = p.eval_grad_f(x0, y0)
        derivs.append(grad_f.dot(dir_Y))

        hessian_f = p.eval_hessian_f(x0, y0)
        d2_f_X = hessian_f.dot(dir_X).dot(dir_X)
        d2_f_Y = hessian_f.dot(dir_Y).dot(dir_Y)

        derivs.append(
            -d2_f_X - k**2 * p.eval_f(x0, y0) + d2_f_Y
        )

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * Y**l

        return v

    def extend_inhomogeneous_radius1(self, x, y):
        dir_X = np.array((1, 0))
        dir_Y = np.array((0, 1))
        return self._extend_inhomogeneous_radius(x, 0, dir_X, dir_Y, y)

    def extend_inhomogeneous_radius2(self, x, y):
        R = self.R
        a = self.a

        dir_X = np.array((np.cos(a), np.sin(a)))
        dir_Y = np.array((np.sin(a), -np.cos(a)))

        x0, y0 = self.get_radius_point(2, x, y) 
        Y = self.signed_dist_to_radius(2, x, y)
        return self._extend_inhomogeneous_radius(
            x0, y0, dir_X, dir_Y, Y)
