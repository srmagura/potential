import numpy as np

class PsGrid:

    def ps_construct_grids(self, scheme_order):
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
            if th >= a:
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
