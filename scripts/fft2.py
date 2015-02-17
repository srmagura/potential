import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy

def eval_phi0(th):
    return np.cos(th)

fourier_N = 1024
th_data = np.linspace(0, 2*np.pi, fourier_N+1)[:-1]
print(th_data[0])
print(th_data[-1])
exact_data = np.array([eval_phi0(th) for th in th_data])

expansion_data = np.zeros(len(th_data))

J_max = 100

coef_raw = np.fft.fft(exact_data)

J_dict = collections.OrderedDict(((J, None) for J in 
    range(-J_max, J_max+1)))

i = 0
coef = np.zeros(len(J_dict), dtype=complex)

for J in J_dict:
    J_dict[J] = i
    coef[i] = coef_raw[J] / fourier_N
    i += 1

for l in range(len(th_data)):
    th = th_data[l]

    phi0 = 0 
    for J, i in J_dict.items():
        exp = np.exp(complex(0, J*th))
        phi0 += (coef[i] * exp).real

    expansion_data[l] = phi0

error = np.max(np.abs(expansion_data - exact_data))

print('---- J_max = {} ----'.format(J_max))
print('Fourier error:', error)

plt.plot(th_data, exact_data, linewidth=5, color='#BBBBBB', label='Exact')
plt.plot(th_data, expansion_data, label='Expansion')
plt.show()
