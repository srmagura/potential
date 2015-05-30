import numpy as np
import itertools as it

s = 2*np.pi
a_cap = 10
bad_k = []

for a in range(1, a_cap):
    for b in range(a, a_cap):
        k = (a*np.pi/s)**2 + (b*np.pi/s)**2
        bad_k.append(k)

print('Auxiliary domain length: {}\n'.format(s))

bad_k.sort()
print('k for which resonance will occur:')
print(bad_k)
