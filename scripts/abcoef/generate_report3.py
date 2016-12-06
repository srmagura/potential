import json
import matplotlib.pyplot as plt
import numpy as np
import itertools as it

_problems = ['trace-hat', 'trace-line-sine', 'trace-parabola']
_boundaries = ['outer-sine', 'sine7', 'inner-sine', 'cubic']
_k = [3.25, 6.75, 10.75]

for problem, boundary, k in it.product(_problems, _boundaries, _k):
    data = json.load(open('m1trend_{}_{}_{}.dat'.format(problem, boundary, k)))

    m1 = np.array(data['m1_data'])

    new_error = []
    singular_vals = []
    for i in range(len(data['error'])):
        new_error.append(np.max(data['error'][i]))
        singular_vals.append(data['singular_vals'][i][-1])

    label = '({}, {})'.format(problem, boundary)
    plt.semilogy(m1, new_error, label=label, markevery=14)

plt.xlabel('$m_1$', fontsize=20)
plt.ylabel('max(error in $a_1$, ..., error in $a_{M_w}$)', fontsize=16)

#plt.legend(loc='upper right')
plt.savefig('/Users/sam/Google Drive/research/output/m1raw.pdf')
plt.show()
