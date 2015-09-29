import numpy as np

def wf1(r):
    b = 3/5
    if r < b:
        return 1.5*np.cos(r*np.pi/b) + 2.5
    else:
        return 1
