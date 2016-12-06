# Plot a_m error vs m1, plotting min, average, and max
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import itertools as it

import seaborn as sns
sns.set_style('whitegrid')

_problems = ['trace-hat', 'trace-line-sine', 'trace-parabola']
_boundaries = ['outer-sine', 'sine7', 'inner-sine', 'cubic']
_k = [3.25, 6.75, 10.75]
ncases = len(_problems) * len(_boundaries) * len(_k)

m1_list = range(10, 276, 5)
allerror = np.zeros((ncases, len(m1_list)))

n = 0
for problem, boundary, k in it.product(_problems, _boundaries, _k):
    data = json.load(open('m1trend_{}_{}_{}.dat'.format(problem, boundary, k)))

    m1 = np.array(data['m1_data'])

    for i in range(len(data['error'])):
        allerror[n, i] = np.max(data['error'][i])

    n += 1

_min = allerror.min(axis=0)
_median = np.median(allerror, axis=0)
_max = allerror.max(axis=0)


minmaxstyle = {
    'linestyle': '--',
    'linewidth': 2,
    'color': '#888888',
}
plt.semilogy(m1_list, _min, label='min', **minmaxstyle)
plt.semilogy(m1_list, _median, linewidth=2, color='#444444', label='median')
plt.semilogy(m1_list, _max, label='max', **minmaxstyle)

plt.fill_between(m1_list, _min, _max, facecolor='gray', alpha=0.2)

plt.text(110, 2.2e-12, 'min', fontsize=14)
plt.text(160, 3e-10, 'median', fontsize=14)
plt.text(210, 2e-7, 'max', fontsize=14)

plt.xlabel('$M_a$', fontsize=20)
plt.ylabel('max(error in $a_1$, ..., error in $a_{M_w}$)', fontsize=16)
sns.despine()
#plt.legend(loc='upper right')
plt.savefig('/Users/sam/Google Drive/research/output/Matrend.pdf')
plt.show()
