import numpy as np

import ps.ps

class ZMethod():

    def __init__(self, options):
        self.problem = options['problem']
        self.scheme_order = options['scheme_order']

    def run(self):
        #TODO
        solver = ps.ps.PizzaSolver({})
        result = solver.run()
