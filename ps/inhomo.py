import numpy as np
import math

from domain_util import cart_to_polar
from .multivalue import Multivalue
from .extend import EType

class PsInhomo:

    # TODO this class is for the arc only.
    # Need to add arbitrary curve extension

    def _extend_f_get_sid(self, i, j):
        R = self.R
        a = self.a

        r, th = self.get_polar(i, j)

        if ((i, j) in self.all_gamma[0] and
            self.get_etype(0, i, j) == EType.standard):
            return 0
        elif r < self.R:
            if th >= 2*np.pi:
                return 1
            else:
                return 2
        else:
            return 0

    def extend_f(self):
        """
        Extend the source term f to grid nodes slightly outside the
        domain.
        """
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
                try:
                    n, th0 = self.boundary_coord_cache[i, j]
                except KeyError:
                    print('Aborting extend_f  !!!!')
                    return

                r0 = self.boundary.eval_r(th0)

                if etype == EType.standard:
                    x0 = r0 * np.cos(th0)
                    y0 = r0 * np.sin(th0)
                elif etype == EType.left:
                    x0 = R * np.cos(a)
                    y0 = R * np.sin(a)
                elif etype == EType.right:
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

            self.f[i,j] = v

    def _extend_inhomo_12_r(self, i, j, radius_sid):
        """
        Do the inhomogeneous 1r and 2r extension

        Use Taylor series to approximate f, grad_f, and hessian_f
        a little bit away from the end of the radius (segment 1 or segment 2)
        """

        p = self.problem
        R = self.R
        a = self.a

        x2, y2 = self.get_coord(i, j)

        if radius_sid == 1:
            x0, y0 = (R, 0)
            x1, y1 = (x2, 0)
            n = y2
        elif radius_sid == 2:
            x0 = R * np.cos(a)
            y0 = R * np.sin(a)

            n = self.signed_dist_to_radius(radius_sid, x2, y2)

            x1, y1 = self.get_radius_point(radius_sid, x2, y2)

        # Direction vector for Taylor series
        tangent = np.array([x1-x0, y1-y0])
        delta = np.linalg.norm(tangent)
        tangent /= np.linalg.norm(tangent)

        f0 = p.eval_f(x0, y0)

        grad_f0 = p.eval_grad_f(x0, y0)
        hessian_f = p.eval_hessian_f(x0, y0)

        f_derivs = (
            f0,
            grad_f0.dot(tangent),
            hessian_f.dot(tangent).dot(tangent)
        )

        f = 0
        for l in range(len(f_derivs)):
            f += f_derivs[l] * delta**l / math.factorial(l)

        grad_f = grad_f0 + hessian_f.dot(tangent) * delta

        # Vector normal to radius, pointing to the destination point
        normal = np.array([x2-x1, y2-y1])

        if np.linalg.norm(normal) != 0:
            normal /= np.linalg.norm(normal)

        d_f_n = grad_f.dot(normal)

        # If normal points in, flip sign of d_f_n
        if n < 0:
            d_f_n *= -1

        d2_f_n = hessian_f.dot(normal).dot(normal)

        d2_f_s = hessian_f.dot(tangent).dot(tangent)

        v = self.inhomo_extend_arbitrary(
            n=n,
            f=f,
            d_f_n=d_f_n,
            d2_f_n=d2_f_n,
            d2_f_s=d2_f_s,
            curv=0,
        )

        return {'elen': delta + abs(n), 'value': v}

    def do_extend_inhomo_0_standard(self, i, j):
        n, th = self.boundary_coord_cache[(i, j)]
        r0 = self.boundary.eval_r(th)

        p = self.problem

        x0 = r0 * np.cos(th)
        y0 = r0 * np.sin(th)

        x1, y1 = self.get_coord(i, j)

        vec = np.array((x1-x0, y1-y0))
        vec /= np.linalg.norm(vec)

        d_f_n = p.eval_grad_f_polar(r0, th).dot(vec) * np.sign(n)

        hessian = p.eval_hessian_f_polar(r0, th)
        d2_f_n = hessian.dot(vec).dot(vec)

        v = self.inhomo_extend_polar(
            n=n,
            f=p.eval_f_polar(r0, th),
            d_f_n=d_f_n,
            d2_f_n=d2_f_n,
            d_f_th=p.eval_d_f_th(r0, th),
            d2_f_th=p.eval_d2_f_th(r0, th),
            d_f_r=p.eval_d_f_r(r0, th),
            d2_f_r=p.eval_d2_f_r(r0, th),
            d2_f_r_th=p.eval_d2_f_r_th(r0, th),
            d_th_s=self.boundary.eval_d_th_s(th),
            d2_th_s=self.boundary.eval_d2_th_s(th),
            d_r_th=self.boundary.eval_d_r_th(th),
            d2_r_th=self.boundary.eval_d2_r_th(th),
            curv=self.boundary.eval_curv(th),
        )

        return {'elen': abs(n), 'value': v}

    def _extend_inhomo_0_lr(self, i, j, radius_sid):
        """
        0l and 0r inhomogeneous extensions

        First use the gradient and Hessian of f to approximate
        f and its derivatives on an extension of the outer boundary (arc).
        Then perform the extension using these approximate derivatives.

        "left" = upper corner
        "right" = lower corner
        """
        p = self.problem
        R = self.R
        a = self.a

        n, th1 = self.boundary_coord_cache[(i, j)]
        r1 = self.boundary.eval_r(th1)

        if radius_sid == 1:
            th0 = 2*np.pi
        elif radius_sid == 2:
            th0 = a

        # Starting point for Taylor
        x0 = R * np.cos(th0)
        y0 = R * np.sin(th0)

        # Ending point for Taylor and starting point for equation-based
        # extension
        x1 = r1 * np.cos(th1)
        y1 = r1 * np.sin(th1)

        # Unit vector in direction of Taylor
        vec = np.array((x1-x0, y1-y0))
        delta = np.linalg.norm(vec)
        vec /= np.linalg.norm(vec)

        # Starting values for Taylor
        f0 = p.eval_f_polar(R, th0)
        grad_f0 = p.eval_grad_f_polar(R, th0)
        hessian_f = hessian_f0 = p.eval_hessian_f_polar(R, th0)

        f_derivs = [f0,
            grad_f0.dot(vec),
            hessian_f0.dot(vec).dot(vec)
        ]

        # Taylor series for f
        f = 0
        for l in range(len(f_derivs)):
            f += f_derivs[l] * delta**l / math.factorial(l)

        # Taylor series for gradient of f
        grad_f = grad_f0 + delta * hessian_f0.dot(vec)

        # Ending point for equation-based extension
        x2, y2 = self.get_coord(i, j)
        r2, th2 = self.get_polar(i, j)

        # First and second normal derivatives of f
        normal = np.array((x2-x1, y2-y1))
        normal /= np.linalg.norm(normal)

        d_f_n = grad_f.dot(normal)

        # If normal points inwards, we need to flip the sign of d_f_n
        if r2 < self.boundary.eval_r(th2):
            d_f_n *= -1

        d2_f_n = hessian_f.dot(normal).dot(normal)

        # Do Taylor for d_f_th
        polar_vec = np.array([r1 - R, th1 - th0])

        # The polar gradient of d_f_th
        d_f_th_grad0 = np.array([
            p.eval_d2_f_r_th(R, th0),
            p.eval_d2_f_th(R, th0)
        ])

        d_f_th = p.eval_d_f_th(R, th0) + d_f_th_grad0.dot(polar_vec)

        # The polar gradient of d_f_r
        d_f_r_grad0 = np.array([
            p.eval_d2_f_r(R, th0),
            p.eval_d2_f_r_th(R, th0)
        ])

        d_f_r = p.eval_d_f_r(R, th0) + d_f_r_grad0.dot(polar_vec)

        v = self.inhomo_extend_polar(
            n=n,
            f=f,
            d_f_n=d_f_n,
            d2_f_n=d2_f_n,
            d_f_th=d_f_th,
            d2_f_th=p.eval_d2_f_th(R, th0),
            d_f_r=d_f_r,
            d2_f_r=p.eval_d2_f_r(R, th0),
            d2_f_r_th=p.eval_d2_f_r_th(R, th0),
            d_th_s=self.boundary.eval_d_th_s(th1),
            d2_th_s=self.boundary.eval_d2_th_s(th1),
            d_r_th=self.boundary.eval_d_r_th(th1),
            d2_r_th=self.boundary.eval_d2_r_th(th1),
            curv=self.boundary.eval_curv(th1),
        )

        return {
            'elen': abs(delta) + abs(n),
            'value': v
        }

    def do_extend_inhomo_0_left(self, i, j):
        return self._extend_inhomo_0_lr(i, j, 2)

    def do_extend_inhomo_0_right(self, i, j):
        return self._extend_inhomo_0_lr(i, j, 1)

    def _calc_inhomo_radius(self, x0, y0, x1, y1, tan, n):
        p = self.problem
        f = p.eval_f(x0, y0)

        if x0 == 0 and y0 == 0:
            '''
            Use Taylor's theorem to construct a smooth extension of
            grad_f, and hessian_f at the origin, since they may
            be undefined or annoying to calculate at this point.
            '''
            h = self.AD_len / (10*self.N)

            hessian_f = p.eval_hessian_f(-h, 0)

            grad_f = p.eval_grad_f(-h, 0)
            grad_f += h * hessian_f.dot((1, 0))
        else:
            grad_f = p.eval_grad_f(x0, y0)
            hessian_f = p.eval_hessian_f(x0, y0)

        vec = np.array((x1-x0, y1-y0))

        if np.linalg.norm(vec) != 0:
            vec /= np.linalg.norm(vec)

        d_f_n = grad_f.dot(vec) * np.sign(n)
        d2_f_n = hessian_f.dot(vec).dot(vec)

        d2_f_s = hessian_f.dot(tan).dot(tan)

        elen = abs(n)

        # Only so that elen matches that of the homogeneous extension
        r, th = cart_to_polar(x0, y0)
        if x0 < 0 or r > self.R:
            elen += r

        v = self.inhomo_extend_arbitrary(
            n=n,
            f=f,
            d_f_n=d_f_n,
            d2_f_n=d2_f_n,
            d2_f_s=d2_f_s,
            curv=0,
        )

        return {
            'elen': elen,
            'value': v
        }

    def _extend_inhomo_radius(self, i, j, radius_sid):
        x1, y1 = self.get_coord(i, j)

        if radius_sid == 1:
            x0, y0 = (x1, 0)
            tan = np.array((-1, 0))
            n = y1
        elif radius_sid == 2:
            x0, y0 = self.get_radius_point(radius_sid, x1, y1)

            a = self.a
            tan = np.array((np.cos(a), np.sin(a)))
            n = self.signed_dist_to_radius(radius_sid, x1, y1)

        return self._calc_inhomo_radius(x0, y0, x1, y1, tan, n)

    def do_extend_inhomo_1_standard(self, i, j):
        return self._extend_inhomo_radius(i, j, 1)

    def do_extend_inhomo_1_left(self, i, j):
        return self._extend_inhomo_radius(i, j, 1)

    def do_extend_inhomo_1_right(self, i, j):
        return self._extend_inhomo_12_r(i, j, 1)

    def do_extend_inhomo_2_standard(self, i, j):
        return self._extend_inhomo_radius(i, j, 2)

    def do_extend_inhomo_2_left(self, i, j):
        return self._extend_inhomo_radius(i, j, 2)

    def do_extend_inhomo_2_right(self, i, j):
        return self._extend_inhomo_12_r(i, j, 2)

    def mv_extend_inhomo_f(self):
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
                    if etype == EType.standard:
                        result = self.do_extend_inhomo_0_standard(i, j)
                    elif etype == EType.left:
                        result = self.do_extend_inhomo_0_left(i, j)
                    elif etype == EType.right:
                        result = self.do_extend_inhomo_0_right(i, j)

                elif sid == 1:
                    if etype == EType.standard:
                        result = self.do_extend_inhomo_1_standard(i, j)
                    elif etype == EType.left:
                        result = self.do_extend_inhomo_1_left(i, j)
                    elif etype == EType.right:
                        result = self.do_extend_inhomo_1_right(i, j)

                elif sid == 2:
                    if etype == EType.standard:
                        result = self.do_extend_inhomo_2_standard(i, j)
                    elif etype == EType.left:
                        result = self.do_extend_inhomo_2_left(i, j)
                    elif etype == EType.right:
                        result = self.do_extend_inhomo_2_right(i, j)

                result['setype'] = (sid, etype)
                mv_ext[(i, j)].append(result)

        return mv_ext

    def extend_inhomo_f(self):
        return self.mv_extend_inhomo_f().reduce()
