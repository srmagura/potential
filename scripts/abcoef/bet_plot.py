import sys

import numpy as np
import matplotlib.pyplot as plt

import json

m = 1

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'bet.dat'

obj = json.load(open(filename))

bet = []
a_error = []
last_sing = []

for datum in obj['results']:
    bet.append(datum['bet'])
    a_error.append(datum['a_error'][m-1])
    last_sing.append(
        datum['singular_vals'][-1]
    )

color1 = '#b3c4ff'
color2 = 'blue'

plt.plot(np.log10(bet), np.log10(a_error), '-', color=color1,
    linewidth=5, label='a_{} error'.format(m))
plt.plot(np.log10(bet), np.log10(last_sing), '--', color=color2,
    linewidth=2, label='smallest sing val')
plt.legend(loc='lower left')


plt.gca().invert_xaxis()

fs = 16
plt.xlabel(r'$\log_{10}(\beta)$', fontsize=fs)
plt.ylabel(r'$\log_{10}$', fontsize=fs)
plt.xlim(-0.25, -3)

#plt.savefig('plotbet_manyk.pdf')
plt.show()
