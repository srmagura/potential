import numpy as np
from ps.extend import EType

class PsGrid:

    # FIXME old way is deprecated
    '''def ps_construct_grids(self, scheme_order):
        self.construct_grids(scheme_order)

        R = self.R # remove eventually?
        a = self.a

        self.all_Mplus = {0: set(), 1: set(), 2: set()}
        self.all_Mminus = {0: set(), 1: set(), 2: set()}
        self.all_gamma = {}

        for i, j in self.M0:
            r, th = self.get_polar(i, j)
            x, y = self.get_coord(i, j)

            boundary_r = self.boundary.eval_r(th)

            # Segment 0
            if th >= a and th <= 2*np.pi:
                if r <= boundary_r:
                    self.all_Mplus[0].add((i, j))
                else:
                    self.all_Mminus[0].add((i, j))

            # Segment 1
            if 0 <= x and x <= R:
                if y <= 0:
                    if r <= boundary_r:
                        self.all_Mplus[1].add((i, j))
                else:
                    self.all_Mminus[1].add((i, j))

            # Segment 2
            x1, y1 = self.get_radius_point(2, x, y)
            dist = self.signed_dist_to_radius(2, x, y)
            if 0 <= x1 and x1 <= R*np.cos(a):
                if dist <= 0:
                    if r <= boundary_r and y >= 0:
                        self.all_Mplus[2].add((i, j))
                else:
                    self.all_Mminus[2].add((i, j))

        union_gamma_set = set()

        for sid in range(3):
            Nplus = set()
            Nminus = set()

            for i, j in self.M0:
                Nm = set([(i, j), (i-1, j), (i+1, j), (i, j-1), (i, j+1)])

                if scheme_order > 2:
                    Nm |= set([(i-1, j-1), (i+1, j-1), (i-1, j+1),
                        (i+1, j+1)])

                if (i, j) in self.all_Mplus[sid]:
                    Nplus |= Nm
                elif (i, j) in self.all_Mminus[sid]:
                    Nminus |= Nm

            gamma_set = Nplus & Nminus
            self.all_gamma[sid] = list(gamma_set)
            union_gamma_set |= gamma_set

        self.union_gamma = list(union_gamma_set)

        if self.fake_grid:
            return self.ps_construct_fake_grid()

    def ps_construct_fake_grid(self):
        """
        For testing extension test only. Dangerous
        """
        R = self.R
        a = self.a
        h = self.AD_len / self.N

        inv = self.get_coord_inv

        self.all_gamma = {
            0: [
                #inv(R+h/2, h/2),
                #inv(-R+h/2, h/2),
                #inv(-R-h/2, 0),
                #inv(R*np.cos(a)+h/2, R*np.sin(a)-h/2),
                #inv(R*np.cos(a)-h/2, R*np.sin(a)-h/2),
                #inv(R*np.cos(a)-h/2, R*np.sin(a)+h/2),
            ],
            1: [
                #inv(R+h/2, h/2),
                #inv(R+h/2, -h/2),
            ],
            2: []#inv(R*np.cos(a)+h/2, R*np.sin(a)-h/2)],
        }

        self.union_gamma = set()
        for sid in range(3):
            self.union_gamma |= set(self.all_gamma[sid])
        self.union_gamma = list(self.union_gamma)

    def ps_grid_dist_test(self):
        def get_dist(node, setype):
            def dformula(x0, y0, _x, _y):
                return np.sqrt((x0-_x)**2 + (y0-_y)**2)

            x, y = self.get_coord(*node)
            dist = -1
            R = self.R
            a = self.a

            if setype == (0, EType.standard):
                n, th = self.boundary.get_boundary_coord(
                    *self.get_polar(*node)
                )
                dist = abs(n)

            elif setype == (0, EType.left):
                x0, y0 = (R*np.cos(a), R*np.sin(a))
                dist = dformula(x0, y0, x, y)

            elif setype == (0, EType.right):
                x0, y0 = (R, 0)
                dist = dformula(x0, y0, x, y)

            elif setype == (1, EType.standard):
                dist = abs(y)

            elif setype == (1, EType.left):
                x0, y0 = (0, 0)
                dist = dformula(x0, y0, x, y)

            elif setype == (1, EType.right):
                x0, y0 = (R, 0)
                dist = dformula(x0, y0, x, y)

            elif setype == (2, EType.standard):
                dist = self.dist_to_radius(2, x, y)

            elif setype == (2, EType.left):
                x0, y0 = (0, 0)
                dist = dformula(x0, y0, x, y)

            elif setype == (2, EType.right):
                x0, y0 = (R*np.cos(a), R*np.sin(a))
                dist = dformula(x0, y0, x, y)

            return dist

        all_gamma2 = {0: set(), 1: set(), 2: set()}

        for node in self.union_gamma:
            for sid in (0, 1, 2):
                etype = self.get_etype(sid, *node)
                setype = (sid, etype)
                dist = get_dist(node, setype)

                h = self.AD_len / self.N
                if dist <= h*np.sqrt(2):
                    all_gamma2[sid].add(node)

        for sid in (0, 1, 2):
            print('=== {} ==='.format(sid))
            diff = all_gamma2[sid] - set(self.all_gamma[sid])
            print('all_gamma2 - all_gamma:', all_gamma2[sid] - set(self.all_gamma[sid]))
            for node in diff:
                print('{}: x={}  y={}'.format(node, *self.get_coord(*node)))

            print('all_gamma - all_gamma2:', set(self.all_gamma[sid]) - all_gamma2[sid])
            print()

        #assert self.all_gamma == all_gamma2
'''

    def ps_construct_grids(self, scheme_order):
        self.construct_grids(scheme_order)

        self.Nplus = set()
        self.Nminus = set()

        for i, j in self.M0:
            Nm = set([(i, j), (i-1, j), (i+1, j), (i, j-1), (i, j+1)])

            if scheme_order > 2:
                Nm |= set([(i-1, j-1), (i+1, j-1), (i-1, j+1),
                    (i+1, j+1)])

            if (i, j) in self.global_Mplus:
                self.Nplus |= Nm
            elif (i, j) in self.global_Mminus:
                self.Nminus |= Nm

        self.union_gamma = list(self.Nplus & self.Nminus)


        def get_dist(node, setype):
            def dformula(x0, y0, _x, _y):
                return np.sqrt((x0-_x)**2 + (y0-_y)**2)

            x, y = self.get_coord(*node)
            dist = -1
            R = self.R
            a = self.a

            if setype == (0, EType.standard):
                n, th = self.boundary.get_boundary_coord(
                    *self.get_polar(*node)
                )
                dist = abs(n)

            elif setype == (0, EType.left):
                x0, y0 = (R*np.cos(a), R*np.sin(a))
                dist = dformula(x0, y0, x, y)

            elif setype == (0, EType.right):
                x0, y0 = (R, 0)
                dist = dformula(x0, y0, x, y)

            elif setype == (1, EType.standard):
                dist = abs(y)

            elif setype == (1, EType.left):
                x0, y0 = (0, 0)
                dist = dformula(x0, y0, x, y)

            elif setype == (1, EType.right):
                x0, y0 = (R, 0)
                dist = dformula(x0, y0, x, y)

            elif setype == (2, EType.standard):
                dist = self.dist_to_radius(2, x, y)

            elif setype == (2, EType.left):
                x0, y0 = (0, 0)
                dist = dformula(x0, y0, x, y)

            elif setype == (2, EType.right):
                x0, y0 = (R*np.cos(a), R*np.sin(a))
                dist = dformula(x0, y0, x, y)

            return dist

        self.all_gamma = {0: [], 1: [], 2: []}

        for node in self.union_gamma:
            r, th = self.get_polar(*node)
            placed = False
            for sid in (0, 1, 2):
                etype = self.get_etype(sid, *node)
                setype = (sid, etype)
                dist = get_dist(node, setype)

                h = self.AD_len / self.N
                if dist <= h*np.sqrt(2):
                    self.all_gamma[sid].append(node)
                    placed = True

            # Every node in union_gamma should go in at least one of the
            # all_gamma sets
            assert placed
