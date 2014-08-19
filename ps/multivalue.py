import numpy as np

import matrices

class Multivalue:

    def __init__(self, solver):
        self.solver = solver
        self.data = {}
        
    def set(self, ij, sid, value):
        if ij not in self.data:
            self.data[ij] = {}
        
        self.data[ij][sid] = value
        
    def has(self, ij):
        return ij in self.data
        
    def get(self, ij, sid=None):
        _dict = self.data[ij]
        
        for sid1, value in _dict.items():
            if sid is None or sid1 == sid:
                return value 
        
    def force_single_value_array(self):
        N = self.solver.N
        array = np.zeros((N-1)**2, dtype=complex)
        
        for i, j in self.data:
            array[matrices.get_index(N, i, j)] = self.get((i, j))
            
        return array
        
    def add_array(self, array):
        N = self.solver.N
        result = Multivalue(self.solver)
        
        for i, j in self.solver.M0:
            index = matrices.get_index(N, i, j)
            
            if (i, j) in self.data:
                _dict = self.data[(i, j)]

                for sid1, value1 in _dict.items():
                    value = value1
                    sid = sid1
            else:
                value = 0
                sid = 0
                        
            result.set((i, j), sid, value + array[index])
                
        return result
        
    def get_interior_array(self):
        N = self.solver.N     
        array = np.zeros((N-1)**2, dtype=complex)
        
        for i, j in self.solver.global_Mplus:           
            index = matrices.get_index(N, i, j)
            x, y = self.solver.get_coord(i, j)
            _dict = self.data[(i, j)]
            
            if len(_dict) == 1:
                array[index] = tuple(_dict.values())[0]
            else:
                if y <= 0:
                    array[index] = _dict[1]
                else:
                    array[index] = _dict[2]

        return array
