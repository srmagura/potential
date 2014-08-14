import numpy as np

class PsGrid:

    def ps_construct_grids(self):
        R = self.R
        a = self.a
    
        self.all_Mplus = {}
        self.all_Mminus = {}
        self.all_gamma = {}
    
        # Segments 1 and 2
        for sid in (1, 2):
            min_x = 0
            
            if sid == 1:
                max_x = R
                slope = 0
            elif sid == 2:
                max_x = R*np.cos(a)
                slope = np.tan(a)
        
            self.all_Mplus[sid] = set()
            self.all_Mminus[sid] = set()
        
            for i, j in self.M0:
                x, y = self.get_coord(i, j)
                
                if min_x <= x and x <= max_x:
                    if((sid == 1 and y <= slope*x) or
                        (sid == 2 and y >= slope*x and not (x == 0 and y == 0))):
                        self.all_Mplus[sid].add((i, j))
                    else:
                        self.all_Mminus[sid].add((i, j))
        
        # Segment 0
        self.all_Mplus[0] = set()
        self.all_Mminus[0] = set()
        
        for i, j in self.M0:
            r, th = self.get_polar(i, j)
            
            if th >= a:
                if r <= self.R:
                    self.all_Mplus[0].add((i, j))
                else:
                    self.all_Mminus[0].add((i, j))
        
        
        for sid in range(3):                
            Nplus = set()
            Nminus = set()
               
            for i, j in self.M0:
                Nm = set([(i, j), (i-1, j), (i+1, j), (i, j-1), (i, j+1)])

                if self.scheme_order > 2:
                    Nm |= set([(i-1, j-1), (i+1, j-1), (i-1, j+1),
                        (i+1, j+1)])

                if (i, j) in self.all_Mplus[sid]:
                    Nplus |= Nm
                elif (i, j) in self.all_Mminus[sid]:
                    Nminus |= Nm

            self.all_gamma[sid] = list(Nplus & Nminus)
