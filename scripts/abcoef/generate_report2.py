import json
import matplotlib.pyplot as plt
import numpy as np

dms = 6.5
data = [
    {'problem': 'trace-line-sine', 'fname': 'LineSine',
        'boundary': 'inner-sine',
        'color': 'blue', 'marker': 'o', 'ms': dms},
    {'problem': 'trace-line-sine', 'fname': 'LineSine',
        'boundary': 'cubic', 'color': 'green', 'marker': '^', 'ms': dms},
    {'problem': 'trace-parabola', 'fname': 'Para',
        'boundary': 'outer-sine', 'color': 'red', 'marker': 'd', 'ms': dms},
    {'problem': 'trace-parabola', 'fname': 'Para',
        'boundary': 'sine7', 'color': 'purple', 'marker': '*', 'ms': dms+2.5},
    {'problem': 'trace-hat', 'fname': 'Hat',
        'boundary': 'outer-sine', 'color': 'black', 'marker': 's', 'ms': dms-.5},
]


for dct in data:
    problem = dct['problem']
    boundary = dct['boundary']

    data = json.load(open('m1trend_{}_{}.dat'.format(problem, boundary)))

    m1 = np.array(data['m1_data'])

    new_error = []
    singular_vals = []
    for i in range(len(data['error'])):
        new_error.append(np.max(data['error'][i]))
        singular_vals.append(data['singular_vals'][i][-1])

    label = '({}, {})'.format(dct['fname'], boundary)
    plt.semilogy(m1, new_error, label=label,
        color=dct['color'], marker=dct['marker'], ms=dct['ms'],
        markevery=14)
    #plt.plot(m1, singular_vals, label='s:{}'.format(boundary))
    plt.xlabel('$m_1$', fontsize=20)
    plt.ylabel('max(error in $a_1$, ..., error in $a_M$)', fontsize=16)

plt.legend(loc='upper right')
plt.savefig('/Users/sam/Google Drive/research/writeup/images/m1trend.pdf')
plt.show()
