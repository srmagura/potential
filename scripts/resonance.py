"""
Print values of k for which a resonance will occur.

Accepts the side length of the square auxiliary domain as a
command-line argument. This argument can be a Python expression and
may include the variable pi. The default is 2*pi.
"""
import numpy as np
import itertools as it
import sys

if len(sys.argv) == 2:
    s_str = sys.argv[1]
else:
    s_str = '2*pi'

pi = np.pi
s = eval(s_str)

a_cap = 10

# Values of k**2 for which a resonance will occur
bad_k2 = []

for a in range(1, a_cap, 2):
    for b in range(a, a_cap, 2):
        print(a, b)
        k2 = (a*np.pi/s)**2 + (b*np.pi/s)**2
        bad_k2.append(k2)

print('Auxiliary domain length: {}\n'.format(s_str))

bad_k2.sort()
print('k for which resonance will occur:')
np.set_printoptions(precision=15)
print(np.sqrt(bad_k2))
